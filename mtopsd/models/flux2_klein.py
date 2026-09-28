"""FLUX.2 [klein] base 9B: Flux2KleinPipeline.

Differences from the Qwen family that matter here:
  * true CFG with a hardcoded "" negative prompt and NO norm rescale of the combined velocity;
  * the timestep shift mu is empirical and depends on the STEP COUNT
    (compute_empirical_mu(seq_len, num_steps)), so the identity-loss sigma sampler uses mu at
    the deployment step count (rollout_steps) and the OPD rollout uses mu at opd_steps;
  * no preferred-resolution snapping: we pin chain geometry to a fixed point of the
    aspect-preserving /16 size formula so repeated edits keep the same size.
"""
from __future__ import annotations

import numpy as np
import torch
from diffusers.pipelines.flux2.pipeline_flux2_klein import compute_empirical_mu, retrieve_timesteps
from diffusers.utils.torch_utils import randn_tensor

from .base import EditModel


def _dims(area, ar):
    w = round((area * ar) ** 0.5)
    h = round((area / ar) ** 0.5)
    return (w // 16) * 16, (h // 16) * 16


def stable_dims(area, ar, max_iter=8):
    """Fixed point of `_dims` so a chain of edits stays size-stable."""
    w, h = _dims(area, ar)
    for _ in range(max_iter):
        w2, h2 = _dims(area, w / h)
        if (w2, h2) == (w, h):
            break
        w, h = w2, h2
    return w, h


class Flux2KleinModel(EditModel):
    TRUE_CFG = 4.0
    NEGATIVE = ""   # the pipeline's hardcoded negative prompt

    def __init__(self, pretrained_path, device="cuda:0"):
        super().__init__(pretrained_path, device)
        self.train_area = 1024 * 1024
        self.deploy_steps = 50   # step count whose mu the identity-loss sigma sampler uses

    def _load_pipeline(self, path):
        from diffusers import Flux2KleinPipeline
        pipe = Flux2KleinPipeline.from_pretrained(path, torch_dtype=torch.bfloat16)
        pipe.to(self.device)
        pipe.set_progress_bar_config(disable=True)
        return pipe

    def set_resolution(self, area):
        self.train_area = int(area)

    # ------------------------------------------------------------------ inference
    @torch.no_grad()
    def generate(self, prompt, image, steps=30, cfg=None, seed=0, autocast=None):
        cfg = self.TRUE_CFG if cfg is None else cfg
        gen = torch.Generator(device=self.device).manual_seed(seed) if seed is not None else None
        img = image.convert("RGB")
        out_w, out_h = stable_dims(self.train_area, img.size[0] / img.size[1])
        if img.size != (out_w, out_h):
            img = self.pipe.image_processor.resize(img, out_h, out_w)
        with self._autocast(autocast):
            out = self.pipe(image=img, prompt=prompt, height=out_h, width=out_w,
                            guidance_scale=cfg, num_inference_steps=steps,
                            num_images_per_prompt=1, generator=gen)
        return out.images[0]

    # ------------------------------------------------------------------ conditioning
    @torch.no_grad()
    def _encode_image(self, img, height, width, dtype):
        """PIL -> (packed BN-normalized reference latents, reference position ids)."""
        pipe = self.pipe
        device = pipe._execution_device
        t = pipe.image_processor.preprocess(
            pipe.image_processor.resize(img, height, width), height, width)
        lat = pipe._encode_vae_image(t.to(device=device, dtype=pipe.vae.dtype), generator=None)
        ids = pipe._prepare_image_ids([lat]).to(device)
        return pipe._pack_latents(lat).to(dtype), ids

    def _output_ids(self, height, width):
        pipe = self.pipe
        vsf2 = pipe.vae_scale_factor * 2
        grid = torch.empty(1, 1, height // vsf2, width // vsf2)
        return pipe._prepare_latent_ids(grid).to(pipe._execution_device)

    # ------------------------------------------------------------------ identity branch
    def identity_loss(self, image, prompt, args):
        self.transformer.train()
        pipe = self.pipe
        device = pipe._execution_device
        fwd = self.ddp_transformer if self.ddp_transformer is not None else self.transformer
        with torch.no_grad():
            image = image.convert("RGB")
            width, height = stable_dims(self.train_area, image.size[0] / image.size[1])
            pe, tids = pipe.encode_prompt(prompt=[prompt], device=device, num_images_per_prompt=1)
            x0, ref_ids = self._encode_image(image, height, width, pe.dtype)
            img_ids = torch.cat([self._output_ids(height, width), ref_ids], dim=1)
            u = torch.sigmoid(torch.normal(mean=args.logit_mean, std=args.logit_std,
                                           size=(1,), device=device))
            mu = compute_empirical_mu(image_seq_len=x0.shape[1], num_steps=self.deploy_steps)
            sigma = pipe.scheduler.time_shift(mu, 1.0, u).to(x0.dtype)
            noise = randn_tensor(x0.shape, generator=None, device=device, dtype=x0.dtype)
            noisy = sigma.view(1, 1, 1) * noise + (1.0 - sigma.view(1, 1, 1)) * x0
            target = (noise - x0).float()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pred = fwd(hidden_states=torch.cat([noisy, x0], dim=1), timestep=sigma, guidance=None,
                       encoder_hidden_states=pe, txt_ids=tids, img_ids=img_ids,
                       joint_attention_kwargs={}, return_dict=False)[0]
        return torch.mean((pred[:, : noisy.size(1)].float() - target) ** 2)

    # ------------------------------------------------------------------ editing branch
    def _opd_setup(self, instruction, clean_image, drifted_image, num_steps, cfg):
        pipe = self.pipe
        device = pipe._execution_device
        sched = pipe.scheduler
        vsf2 = pipe.vae_scale_factor * 2
        clean_image = clean_image.convert("RGB")
        drifted_image = drifted_image.convert("RGB")
        width, height = stable_dims(self.train_area, clean_image.size[0] / clean_image.size[1])
        seq = (height // vsf2) * (width // vsf2)

        with torch.no_grad():
            pe, tids = pipe.encode_prompt(prompt=[instruction], device=device,
                                          num_images_per_prompt=1)
            npe, ntids = pipe.encode_prompt(prompt=[self.NEGATIVE], device=device,
                                            num_images_per_prompt=1)
            s_el, s_ids = self._encode_image(drifted_image, height, width, pe.dtype)
            t_el, t_ids = self._encode_image(clean_image, height, width, pe.dtype)
            out_ids = self._output_ids(height, width)
        S = dict(el=s_el, ids=torch.cat([out_ids, s_ids], dim=1))   # drifted rollout state
        T = dict(el=t_el, ids=torch.cat([out_ids, t_ids], dim=1))   # clean source

        def v(latents, ts, C, embeds, txt_ids):
            return self.transformer(hidden_states=torch.cat([latents, C["el"]], dim=1),
                                    timestep=ts, guidance=None, encoder_hidden_states=embeds,
                                    txt_ids=txt_ids, img_ids=C["ids"],
                                    joint_attention_kwargs={}, return_dict=False)[0][:, :seq]

        def cfg_v(latents, ts, C):
            with self._autocast():
                pos = v(latents, ts, C, pe, tids)
                with torch.no_grad():
                    neg = v(latents, ts, C, npe, ntids)
            return neg + cfg * (pos - neg)

        timesteps, _ = retrieve_timesteps(sched, num_steps, device,
                                          sigmas=np.linspace(1.0, 1 / num_steps, num_steps),
                                          mu=compute_empirical_mu(image_seq_len=seq,
                                                                  num_steps=num_steps))
        sched.set_begin_index(0)
        return dict(cfg_v=cfg_v, student=S, teacher=T, timesteps=timesteps,
                    sigmas=sched.sigmas, device=device, dtype=S["el"].dtype,
                    latent_shape=(1, seq, self.transformer.config.in_channels))
