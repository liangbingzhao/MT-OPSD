"""Multi-turn image editing with an MT-OPSD checkpoint (or a base model).

    python inference.py --ckpt /path/to/mtopsd-qwen-image-edit-2511 \
        --image assets/example.png --output_dir outputs/example \
        --instructions "Turn the sky into a sunset." "Add a red kite in the sky." "Convert to a watercolor painting."

    # instructions from a text file (one per line), base model for comparison
    python inference.py --model qwen-image-edit-2511 --image in.png --instructions_file turns.txt --output_dir outputs/base

Each turn edits the previous turn's output. Outputs: turn1.png ... turnN.png and strip.png.
"""
from __future__ import annotations

import argparse
import os

from PIL import Image

from mtopsd.inference import load_model, edit_session
from mtopsd.models import MODELS


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ckpt", default=None, help="MT-OPSD checkpoint dir (LoRA).")
    ap.add_argument("--model", default=None, choices=list(MODELS),
                    help="backbone for a base-model run (ignored with --ckpt).")
    ap.add_argument("--pretrained_path", default=None,
                    help="local path / HF id of the base model (default: the backbone's HF id).")
    ap.add_argument("--image", required=True)
    ap.add_argument("--instructions", nargs="+", default=None)
    ap.add_argument("--instructions_file", default=None, help="one instruction per line.")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--steps", type=int, default=None, help="default: the checkpoint's (30).")
    ap.add_argument("--cfg", type=float, default=None, help="default: the checkpoint's (4.0).")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train_area", type=int, default=None,
                    help="working pixel area; default: the checkpoint's (512*512). Keep it equal "
                         "to the training area.")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    if args.instructions_file:
        with open(args.instructions_file) as f:
            instructions = [line.strip() for line in f if line.strip()]
    else:
        instructions = args.instructions
    if not instructions:
        ap.error("give --instructions or --instructions_file")
    if not args.ckpt and not args.model:
        ap.error("give --ckpt or --model")

    model, cfg = load_model(args.ckpt, args.model, args.pretrained_path, args.device,
                            args.train_area)
    steps = args.steps or cfg["steps"]
    guidance = args.cfg if args.cfg is not None else cfg["cfg"]
    os.makedirs(args.output_dir, exist_ok=True)
    src = Image.open(args.image).convert("RGB")

    def save(t, img):
        img.save(os.path.join(args.output_dir, f"turn{t + 1}.png"))
        print(f"[turn {t + 1}/{len(instructions)}] {instructions[t]}", flush=True)

    outs = edit_session(model, src, instructions, steps=steps, cfg=guidance, seed=args.seed,
                        save_fn=save)
    w, h = outs[0].size
    strip = Image.new("RGB", (w * (len(outs) + 1), h), "white")
    for i, im in enumerate([src.resize((w, h))] + outs):
        strip.paste(im.resize((w, h)), (i * w, 0))
    strip.save(os.path.join(args.output_dir, "strip.png"))
    print(f"[done] -> {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
