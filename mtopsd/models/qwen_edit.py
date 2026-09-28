"""Qwen-Image-Edit family (Qwen-Image-Edit-2511, FireRed-Image-Edit-1.0): QwenImageEditPlusPipeline."""
from __future__ import annotations

import numpy as np
import torch

# The working pixel area lives in ONE place, the pipeline module's VAE_IMAGE_SIZE global:
# set_resolution() reassigns it, which retargets our loss geometry, generate()'s output size
# AND the pipeline's internal condition preprocessing (which no call kwarg can override).
import diffusers.pipelines.qwenimage.pipeline_qwenimage_edit_plus as _qpipe
from diffusers.pipelines.qwenimage.pipeline_qwenimage_edit_plus import (
    CONDITION_IMAGE_SIZE,
    calculate_dimensions,
    calculate_shift,
    retrieve_timesteps,
)
from diffusers.utils.torch_utils import randn_tensor

from .base import EditModel


class QwenEditModel(EditModel):
    TRUE_CFG = 4.0
    NEGATIVE = " "

    def _load_pipeline(self, path):
        from diffusers import QwenImageEditPlusPipeline
        pipe = QwenImageEditPlusPipeline.from_pretrained(path, torch_dtype=torch.bfloat16)
        pipe.to(self.device)
        pipe.set_progress_bar_config(disable=True)
        return pipe

    def set_resolution(self, area):
        _qpipe.VAE_IMAGE_SIZE = int(area)

    def _shift_mu(self, seq_len):
        c = self.pipe.scheduler.config
        return calculate_shift(seq_len, c.get("base_image_seq_len", 256),
                               c.get("max_image_seq_len", 4096),
                               c.get("base_shift", 0.5), c.get("max_shift", 1.15))

    # ------------------------------------------------------------------ inference
    @torch.no_grad()
    def generate(self, prompt, image, steps=30, cfg=None, seed=0, autocast=None):
        """One edit of `image`; output at the working area, aspect preserved."""
        cfg = self.TRUE_CFG if cfg is None else cfg
        gen = torch.Generator(device=self.device).manual_seed(seed) if seed is not None else None
        img = image.convert("RGB")
        out_w, out_h = calculate_dimensions(_qpipe.VAE_IMAGE_SIZE, img.size[0] / img.size[1])
        with self._autocast(autocast):
            out = self.pipe(image=[img], prompt=prompt, negative_prompt=self.NEGATIVE,
                            height=out_h, width=out_w, guidance_scale=1.0,
                            num_images_per_prompt=1, generator=gen,
                            true_cfg_scale=cfg, num_inference_steps=steps)
        return out.images[0]

    # ------------------------------------------------------------------ conditioning
    def _geometry(self, ratio):
        vsf2 = self.pipe.vae_scale_factor * 2
        calc_w, calc_h = calculate_dimensions(_qpipe.VAE_IMAGE_SIZE, ratio)
        return calc_w // vsf2 * vsf2, calc_h // vsf2 * vsf2

    @torch.no_grad()
    def _encode(self, img, prompts, ratio, width, height):
        """Prompt embeds (one per prompt, VL branch sees the 384^2 view of `img`) and the
        packed VAE latents of `img` (the condition stream), as in pipe.__call__."""
        pipe = self.pipe
        device = pipe._execution_device
        vsf2 = pipe.vae_scale_factor * 2
        cond_w, cond_h = calculate_dimensions(CONDITION_IMAGE_SIZE, ratio)
        vae_w, vae_h = calculate_dimensions(_qpipe.VAE_IMAGE_SIZE, ratio)
        cond_img = pipe.image_processor.resize(img, cond_h, cond_w)
        embeds = [pipe.encode_prompt(image=[cond_img], prompt=[p], device=device,
                                     num_images_per_prompt=1) for p in prompts]
        vae_img = pipe.image_processor.preprocess(img, vae_h, vae_w).unsqueeze(2)
        _, el = pipe.prepare_latents([vae_img], 1, self.transformer.config.in_channels // 4,
                                     height, width, embeds[0][0].dtype, device, None, None)
        shapes = [[(1, height // vsf2, width // vsf2), (1, vae_h // vsf2, vae_w // vsf2)]]
        return embeds, el, shapes

    # ------------------------------------------------------------------ identity branch
    def identity_loss(self, image, prompt, args):
        """Single-step flow matching with target == condition == `image`."""
        self.transformer.train()
        pipe = self.pipe
        device = pipe._execution_device
        fwd = self.ddp_transformer if self.ddp_transformer is not None else self.transformer
        with torch.no_grad():
            image = image.convert("RGB")
            ratio = image.size[0] / image.size[1]
            width, height = self._geometry(ratio)
            [(pe, pm)], x0, shapes = self._encode(image, [prompt], ratio, width, height)
            u = torch.sigmoid(torch.normal(mean=args.logit_mean, std=args.logit_std,
                                           size=(1,), device=device))
            sigma = pipe.scheduler.time_shift(self._shift_mu(x0.shape[1]), 1.0, u).to(x0.dtype)
            noise = randn_tensor(x0.shape, generator=None, device=device, dtype=x0.dtype)
            noisy = sigma.view(1, 1, 1) * noise + (1.0 - sigma.view(1, 1, 1)) * x0
            target = (noise - x0).float()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pred = fwd(hidden_states=torch.cat([noisy, x0], dim=1), timestep=sigma,
                       guidance=None, encoder_hidden_states=pe, encoder_hidden_states_mask=pm,
                       img_shapes=shapes, attention_kwargs={}, return_dict=False)[0]
        return torch.mean((pred[:, : noisy.size(1)].float() - target) ** 2)

    # ------------------------------------------------------------------ editing branch
    def _opd_setup(self, instruction, clean_image, drifted_image, num_steps, cfg):
        pipe = self.pipe
        device = pipe._execution_device
        sched = pipe.scheduler
        vsf2 = pipe.vae_scale_factor * 2
        clean_image = clean_image.convert("RGB")
        drifted_image = drifted_image.convert("RGB")
        ratio = clean_image.size[0] / clean_image.size[1]
        width, height = self._geometry(ratio)
        seq = (height // vsf2) * (width // vsf2)

        def cond(img):
            [(pe, pm), (npe, npm)], el, shapes = self._encode(
                img, [instruction, self.NEGATIVE], ratio, width, height)
            return dict(el=el, pe=pe, pm=pm, npe=npe, npm=npm, shapes=shapes)

        S = cond(drifted_image)   # student sees the drifted rollout state
        T = cond(clean_image)     # teacher sees the clean source

        def cfg_v(latents, ts, C):
            """CFG velocity as in pipe.__call__ (combine + norm rescale to the cond branch);
            only the cond branch keeps a graph."""
            mi = torch.cat([latents, C["el"]], dim=1)
            kw = dict(hidden_states=mi, timestep=ts, guidance=None, img_shapes=C["shapes"],
                      attention_kwargs={}, return_dict=False)
            with self._autocast():
                pos = self.transformer(encoder_hidden_states=C["pe"],
                                       encoder_hidden_states_mask=C["pm"], **kw)[0][:, :seq]
                with torch.no_grad():
                    neg = self.transformer(encoder_hidden_states=C["npe"],
                                           encoder_hidden_states_mask=C["npm"], **kw)[0][:, :seq]
            comb = neg + cfg * (pos - neg)
            return comb * (pos.norm(dim=-1, keepdim=True) / comb.norm(dim=-1, keepdim=True))

        timesteps, _ = retrieve_timesteps(sched, num_steps, device,
                                          sigmas=np.linspace(1.0, 1 / num_steps, num_steps),
                                          mu=self._shift_mu(seq))
        sched.set_begin_index(0)
        return dict(cfg_v=cfg_v, student=S, teacher=T, timesteps=timesteps,
                    sigmas=sched.sigmas, device=device, dtype=S["el"].dtype,
                    latent_shape=(1, seq, self.transformer.config.in_channels))
