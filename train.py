"""MT-OPSD training entry point.

    accelerate launch --num_processes 4 train.py --config configs/qwen_image_edit_2511.yaml

Any YAML key can be overridden on the command line (e.g. --output_dir, --pretrained_path).
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import timedelta

import torch
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs, InitProcessGroupKwargs
from PIL import Image

from mtopsd.config import add_train_args, parse_with_config
from mtopsd.data import build_dataloader
from mtopsd.models import build_model
from mtopsd.models.base import seed_everything
from mtopsd.trainer import maybe_resume, run_training


def main():
    parser = add_train_args(argparse.ArgumentParser(description=__doc__.split("\n")[0]))
    args = parse_with_config(parser, sys.argv[1:])
    for key in ("model", "data_root", "output_dir"):
        if not getattr(args, key):
            parser.error(f"--{key} is required (in the YAML or on the command line)")

    accelerator = Accelerator(
        mixed_precision="no",   # bf16 autocast is applied inside the model wrappers
        kwargs_handlers=[
            # the identity prompt leaves some text-stream LoRA modules unused; both branches
            # are chosen in lockstep across ranks, so the unused set always matches
            DistributedDataParallelKwargs(find_unused_parameters=True, broadcast_buffers=False),
            InitProcessGroupKwargs(timeout=timedelta(minutes=60)),   # long drift-eval barriers
        ])
    args.device = str(accelerator.device)
    args.proc = accelerator.process_index
    args.world = accelerator.num_processes

    seed_everything(args.seed + args.proc)   # per-rank rollout noise
    model = build_model(args.model, args.pretrained_path, args.device)
    model.set_resolution(args.train_area)
    model.deploy_steps = args.rollout_steps  # FLUX.2: mu of the identity-loss sigma sampler
    trainable = model.add_lora(args.rank, args.lora_alpha, args.target_modules)
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, betas=(args.adam_beta1, args.adam_beta2),
                                  eps=args.adam_eps, weight_decay=args.weight_decay)
    # DDP wraps a separate view used only for the identity-loss forward; model.transformer
    # stays the raw module for sampling, adapter toggling and state I/O
    model.ddp_transformer, optimizer = accelerator.prepare(model.transformer, optimizer)
    loader = build_dataloader(args, rank=args.proc)
    start_step, wandb_id, edit_rng, curriculum = maybe_resume(args, model, optimizer)
    accelerator.wait_for_everyone()

    eval_img = None
    if args.eval_img:
        if os.path.isfile(args.eval_img):
            eval_img = Image.open(args.eval_img).convert("RGB")
        else:
            print(f"[warn] eval_img {args.eval_img} not found; drift eval disabled", flush=True)

    run_training(args, model, loader, optimizer, trainable, accelerator, eval_img=eval_img,
                 start_step=start_step, wandb_run_id=wandb_id, edit_rng_state=edit_rng,
                 curriculum_state=curriculum)


if __name__ == "__main__":
    main()
