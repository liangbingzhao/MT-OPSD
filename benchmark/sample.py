"""Generate LME-Bench sessions with an MT-OPSD checkpoint or a base model.

    python benchmark/sample.py --ckpt /path/to/checkpoint --out outputs/lme/mtopsd_qwen
    python benchmark/sample.py --model qwen-image-edit-2511 --out outputs/lme/qwen_base
    python benchmark/sample.py --model flux2-klein-base-9b --steps 30 --out outputs/lme/flux_base

Writes <out>/<id>/{source.png, turn1.png ... turn10.png}. Rerunning resumes. To split the
100 sessions over several GPUs, launch one process per GPU with --shard i --num_shards N
(same --out).
"""
from __future__ import annotations

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from mtopsd.inference import run_sessions  # noqa: E402
from mtopsd.models import MODELS  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ckpt", default=None, help="MT-OPSD checkpoint dir.")
    ap.add_argument("--model", default=None, choices=list(MODELS),
                    help="backbone for a base-model run (ignored with --ckpt).")
    ap.add_argument("--pretrained_path", default=None)
    ap.add_argument("--data_dir", default=os.path.join(HERE, "data"),
                    help="downloaded LME-Bench dir holding images/<id>.png.")
    ap.add_argument("--sessions", default=os.path.join(HERE, "lme_bench.json"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--train_area", type=int, default=None,
                    help="working pixel area; default: the checkpoint's, 512*512 for base models.")
    ap.add_argument("--steps", type=int, default=None, help="default: the checkpoint's, else 30.")
    ap.add_argument("--cfg", type=float, default=None, help="default: the checkpoint's, else 4.0.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num_shards", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0, help="first N sessions only.")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    if not args.ckpt and not args.model:
        ap.error("give --ckpt or --model")

    with open(args.sessions) as f:
        items = json.load(f)["items"]
    if args.limit:
        items = items[: args.limit]
    items = items[args.shard::args.num_shards]
    run_sessions(items, os.path.join(args.data_dir, "images"), args.out, ckpt=args.ckpt,
                 model=args.model, pretrained_path=args.pretrained_path, device=args.device,
                 train_area=args.train_area, steps=args.steps, cfg=args.cfg, seed=args.seed,
                 tag="lme")


if __name__ == "__main__":
    main()
