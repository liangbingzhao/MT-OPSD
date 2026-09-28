"""Backbone-agnostic part of an MT-OPSD model wrapper.

A wrapper holds one diffusers editing pipeline whose DiT carries up to two LoRA adapters:
  "default" - the trainable student (fp32 parameters, bf16 autocast forward)
  "teacher" - an optional frozen adapter; before the first gated promotion the teacher is
              the bare base model (all adapters disabled)

Subclasses implement the backbone-specific pieces:
  _load_pipeline(path), set_resolution(area), generate(...),
  identity_loss(image, prompt, args) -> scalar loss with grad,
  _opd_setup(instruction, clean, drifted, num_steps, cfg) -> dict (see compute_opd_loss).
"""
from __future__ import annotations

import contextlib
import os
import random

import torch

TEACHER_ADAPTER = "teacher"


def sample_query_indices(n_states, k, bias, rng):
    """k distinct rollout indices in [0, n_states); index 0 is pure noise, n-1 near clean.
    low_t = Beta(5,2), mid_t = Beta(5,5), high_t = Beta(2,5) over the normalized index."""
    k = max(1, min(int(k), n_states))
    if k >= n_states:
        return list(range(n_states))
    if bias == "uniform":
        return sorted(rng.sample(range(n_states), k))
    alpha, beta = {"high_t": (2.0, 5.0), "mid_t": (5.0, 5.0), "low_t": (5.0, 2.0)}[bias]
    chosen = set()
    attempts = 0
    while len(chosen) < k and attempts < 50 * k:
        chosen.add(min(int(rng.betavariate(alpha, beta) * n_states), n_states - 1))
        attempts += 1
    while len(chosen) < k:
        chosen.add(rng.randrange(n_states))
    return sorted(chosen)


def lora_state_dict_from_dir(ckpt_dir):
    """peft-format LoRA state dict of a checkpoint dir (training_state.pt or the
    diffusers-format pytorch_lora_weights.safetensors)."""
    ts = os.path.join(ckpt_dir, "training_state.pt")
    if os.path.isfile(ts):
        return torch.load(ts, map_location="cpu", weights_only=False)["lora"]
    from diffusers.utils import convert_unet_state_dict_to_peft
    from safetensors.torch import load_file
    raw = load_file(os.path.join(ckpt_dir, "pytorch_lora_weights.safetensors"))
    return convert_unet_state_dict_to_peft(
        {k[len("transformer."):] if k.startswith("transformer.") else k: v for k, v in raw.items()})


class EditModel:
    TRUE_CFG = 4.0

    def __init__(self, pretrained_path, device="cuda:0"):
        self.device = device
        self.pipe = self._load_pipeline(pretrained_path)
        self.transformer = self.pipe.transformer
        # pure bf16 until a trainable fp32 LoRA is added (then autocast is needed)
        self.use_autocast = False
        # DDP-wrapped view of self.transformer, used only for the identity-loss forward
        self.ddp_transformer = None
        self.teacher_lora = None      # checkpoint dir currently loaded as the teacher
        self.gate_teacher_step = -1   # global step of the promoted teacher (-1 = base)

    def _load_pipeline(self, path):
        raise NotImplementedError

    def _autocast(self, on=None):
        on = self.use_autocast if on is None else on
        return torch.autocast("cuda", dtype=torch.bfloat16) if on else contextlib.nullcontext()

    # ------------------------------------------------------------------ inference
    @torch.no_grad()
    def rollout(self, src_image, rounds, steps, cfg, noop_prompt):
        """`rounds` identity edits of the current student on its own output (fresh noise)."""
        self.transformer.eval()
        cur = src_image.convert("RGB")
        for _ in range(rounds):
            cur = self.generate(noop_prompt, cur, steps=steps, cfg=cfg, seed=None)
        return cur

    @contextlib.contextmanager
    def teacher_ctx(self):
        """Run the enclosed forwards as the teacher, then restore the student."""
        if self.teacher_lora is None:
            self.transformer.disable_adapters()
            try:
                yield
            finally:
                self.transformer.enable_adapters()
        else:
            self.transformer.set_adapter(TEACHER_ADAPTER)
            try:
                yield
            finally:
                self.transformer.set_adapter("default")

    # ------------------------------------------------------------------ losses
    def compute_noop_loss(self, image, args):
        """Identity branch: the rollout state is its own target under e_id."""
        return self.identity_loss(image, args.noop_prompt, args)

    def compute_opd_loss(self, instruction, clean_image, drifted_image, num_steps, cfg,
                         k, query_bias, query_rng):
        """Editing branch: sparse on-policy velocity matching.

        The student (LoRA, conditioned on the drifted rollout state) samples its own
        num_steps-step CFG trajectory; at k sampled states the frozen teacher, conditioned
        on the clean source, predicts the CFG velocity at the same latent and timestep, and
        the student velocity is regressed onto it. The trajectory is advanced with the
        detached student velocity (no BPTT) and each query is back-propagated immediately,
        so memory is O(1) in num_steps. Gradients are all-reduced manually (the student
        forward bypasses DDP, which allows only one backward per step). Returns a float.
        """
        self.transformer.train()
        o = self._opd_setup(instruction, clean_image, drifted_image, num_steps, cfg)
        cfg_v, S, T = o["cfg_v"], o["student"], o["teacher"]
        timesteps, sigmas = o["timesteps"], o["sigmas"]
        n = len(timesteps)
        query_idx = sample_query_indices(n, k, query_bias, query_rng)
        last = query_idx[-1]

        latents = torch.randn(*o["latent_shape"], device=o["device"], dtype=o["dtype"])
        total = 0.0
        for i, t in enumerate(timesteps):
            latents = latents.detach()
            ts = (t / 1000.0).expand(latents.shape[0]).to(latents.dtype)
            dt = (sigmas[i + 1] - sigmas[i]).to(latents.dtype)
            if i in query_idx:
                v_s = cfg_v(latents, ts, S)                  # grad flows through the pos branch
                with self.teacher_ctx(), torch.no_grad():
                    v_t = cfg_v(latents, ts, T)
                loss_i = torch.mean((v_s.float() - v_t.float()) ** 2)
                (loss_i / len(query_idx)).backward()
                total += loss_i.item()
                v_adv = v_s.detach()
            else:
                with torch.no_grad():
                    v_adv = cfg_v(latents, ts, S)
            if i >= last:                                    # the tail is never queried
                break
            latents = latents + dt * v_adv
        self.allreduce_grads()
        return total / len(query_idx)

    def allreduce_grads(self):
        import torch.distributed as dist
        if not (dist.is_available() and dist.is_initialized()):
            return
        world = dist.get_world_size()
        for p in self.transformer.parameters():
            if not p.requires_grad:
                continue
            if p.grad is None:
                p.grad = torch.zeros_like(p)
            dist.all_reduce(p.grad)
            p.grad /= world

    # ------------------------------------------------------------------ LoRA
    def add_lora(self, rank, lora_alpha, target_modules, gradient_checkpointing=True):
        """Freeze the backbone, add the fp32 student adapter; return its parameters."""
        from diffusers.training_utils import cast_training_params
        from peft import LoraConfig

        self.transformer.requires_grad_(False)
        self.transformer.add_adapter(LoraConfig(
            r=rank, lora_alpha=lora_alpha, lora_dropout=0.0,
            init_lora_weights="gaussian", target_modules=list(target_modules)))
        cast_training_params(self.transformer, dtype=torch.float32)
        if gradient_checkpointing:
            # diffusers' default checkpoint function is non-reentrant, which is DDP-safe
            self.transformer.enable_gradient_checkpointing()
        trainable = [p for p in self.transformer.parameters() if p.requires_grad]
        if not trainable:
            raise RuntimeError("no trainable LoRA params; check target_modules.")
        n_train = sum(p.numel() for p in trainable)
        n_total = sum(p.numel() for p in self.transformer.parameters())
        print(f"[lora] trainable params: {n_train:,} / {n_total:,} "
              f"({100 * n_train / n_total:.4f}%)", flush=True)
        self.use_autocast = True
        return trainable

    def load_lora(self, ckpt_dir, rank, lora_alpha, target_modules):
        """Inference: the fp32 adapter + bf16 autocast exactly as during training
        (casting the LoRA to bf16 measurably hurts multi-turn robustness)."""
        from peft import set_peft_model_state_dict
        self.add_lora(rank, lora_alpha, target_modules, gradient_checkpointing=False)
        res = set_peft_model_state_dict(self.transformer, lora_state_dict_from_dir(ckpt_dir))
        if getattr(res, "unexpected_keys", None):
            raise ValueError(f"{len(res.unexpected_keys)} unexpected LoRA keys, e.g. "
                             f"{res.unexpected_keys[:3]} ({ckpt_dir})")
        self.transformer.eval()

    def promote_teacher(self, ckpt_dir, rank, lora_alpha, target_modules):
        """Make checkpoint `ckpt_dir` the frozen teacher. The first promotion installs the
        teacher adapter, later ones overwrite its weights in place (same parameter objects,
        so optimizer / DDP / all-reduce, which all filter on requires_grad, are untouched)."""
        from peft import LoraConfig, set_peft_model_state_dict

        sd = lora_state_dict_from_dir(ckpt_dir)
        if self.teacher_lora is None:
            self.transformer.add_adapter(
                LoraConfig(r=rank, lora_alpha=lora_alpha, lora_dropout=0.0,
                           init_lora_weights="gaussian", target_modules=list(target_modules)),
                adapter_name=TEACHER_ADAPTER)
        res = set_peft_model_state_dict(self.transformer, sd, adapter_name=TEACHER_ADAPTER)
        if getattr(res, "unexpected_keys", None):
            raise ValueError(f"teacher LoRA has {len(res.unexpected_keys)} unexpected keys "
                             f"({ckpt_dir})")
        for name, p in self.transformer.named_parameters():
            if f".{TEACHER_ADAPTER}." in name:
                p.requires_grad_(False)
        self.transformer.set_adapter("default")   # add_adapter activates the new adapter
        self.teacher_lora = ckpt_dir
        print(f"[teacher] <- {ckpt_dir}", flush=True)

    def save_lora(self, output_dir):
        """Student adapter in diffusers format (pipe.load_lora_weights-compatible)."""
        from diffusers.utils import convert_state_dict_to_diffusers
        from peft.utils import get_peft_model_state_dict

        os.makedirs(output_dir, exist_ok=True)
        sd = convert_state_dict_to_diffusers(get_peft_model_state_dict(self.transformer))
        self.pipe.__class__.save_lora_weights(output_dir, transformer_lora_layers=sd,
                                              safe_serialization=True)


def seed_everything(seed):
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
