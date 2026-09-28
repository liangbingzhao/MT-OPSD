"""Export a training checkpoint as a release checkpoint (LoRA weights + adapter_config.json,
without the optimizer state).

    python scripts/export_checkpoint.py RUN/checkpoint-1488 release/mtopsd-qwen-image-edit-2511
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mtopsd.inference import read_checkpoint_config  # noqa: E402
from mtopsd.models import ALIASES  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("ckpt")
    ap.add_argument("out")
    args = ap.parse_args()
    cfg = read_checkpoint_config(args.ckpt)
    cfg["model"] = ALIASES.get(cfg["model"], cfg["model"])
    os.makedirs(args.out, exist_ok=True)
    shutil.copy2(os.path.join(args.ckpt, "pytorch_lora_weights.safetensors"), args.out)
    with open(os.path.join(args.out, "adapter_config.json"), "w") as f:
        json.dump(cfg, f, indent=1)
    print(f"[export] {args.ckpt} -> {args.out}: {cfg}")


if __name__ == "__main__":
    main()
