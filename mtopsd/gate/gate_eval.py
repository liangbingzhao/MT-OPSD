"""Evaluate one model (base or checkpoint) on the gate set: generate the ten-turn sessions,
then judge them with GPT-4o. Resumable at turn and session granularity.

    python -m mtopsd.gate.gate_eval --ckpt RUN/checkpoint-480 --out RUN/gate/evals/step_480
    python -m mtopsd.gate.gate_eval --model qwen-image-edit-2511 --out RUN/gate/evals/base

Output: <out>/<id>/{source,turn1..turn10}.png and <out>/judge/results.json
"""
from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor

from mtopsd import judge
from mtopsd.inference import run_sessions

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
GATE_SET = os.path.join(REPO, "assets", "gate_set", "gate_set.json")


def load_items(path=GATE_SET, shard=0, num_shards=1):
    with open(path) as f:
        items = json.load(f)["items"]
    return items[shard::num_shards]


def run_judge(args, items):
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("[gate_eval] the judge needs OPENAI_API_KEY")
    res_dir = os.path.join(args.out, "judge")
    cache = os.path.join(res_dir, "per_sample")
    os.makedirs(cache, exist_ok=True)
    per_sample = {}
    for it in items:
        p = os.path.join(cache, f"{it['id']}.json")
        if os.path.isfile(p):
            with open(p) as f:
                per_sample[it["id"]] = json.load(f)

    def one(it):
        d = os.path.join(args.out, it["id"])
        paths = [os.path.join(d, "source.png")] + [os.path.join(d, f"turn{t + 1}.png")
                                                   for t in range(len(it["turns"]))]
        r = judge.score_session((api_key, args.api_model), it["turns"], paths)
        tmp = os.path.join(cache, f"{it['id']}.json.tmp")
        with open(tmp, "w") as f:
            json.dump(r, f, indent=1)
        os.replace(tmp, os.path.join(cache, f"{it['id']}.json"))
        return it["id"], r

    todo = [it for it in items if it["id"] not in per_sample]
    print(f"[gate_eval] judging {len(todo)} sessions ({len(per_sample)} cached)", flush=True)
    with ThreadPoolExecutor(max_workers=args.num_workers) as ex:
        for sid, r in ex.map(one, todo):
            per_sample[sid] = r
    errors = [sid for sid, r in per_sample.items() if "error" in r]
    if errors:
        # a silent judge failure (quota, outage) would look like a genuinely bad model
        raise SystemExit(f"[gate_eval] judge failed on {errors}; fix the API and rerun "
                         "(finished sessions are cached)")
    n_turn = len(items[0]["turns"])
    out = {"total": len(items), "done": len(per_sample), "api_model": args.api_model,
           "report": judge.aggregate(per_sample, n_turn), "per_sample": per_sample}
    tmp = os.path.join(res_dir, "results.json.tmp")
    with open(tmp, "w") as f:
        json.dump(out, f, indent=1)
    os.replace(tmp, os.path.join(res_dir, "results.json"))
    sr = out["report"]["success_rate"]
    print(f"[gate_eval] SR@5/8/10 = {sr['turn5']:.3f}/{sr['turn8']:.3f}/{sr['turn10']:.3f} "
          f"-> {res_dir}/results.json", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ckpt", default=None, help="checkpoint dir; omit for the base model.")
    ap.add_argument("--model", default=None, help="backbone name for a base-model eval.")
    ap.add_argument("--pretrained_path", default=None)
    ap.add_argument("--train_area", type=int, default=None)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--cfg", type=float, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", required=True)
    ap.add_argument("--gate_set", default=GATE_SET)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num_shards", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0, help="first N sessions only (smoke tests).")
    ap.add_argument("--api_model", default="gpt-4o")
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--skip_judge", action="store_true")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    items = load_items(args.gate_set, args.shard, args.num_shards)
    if args.limit:
        items = items[: args.limit]
    run_sessions(items, os.path.join(os.path.dirname(args.gate_set), "images"), args.out,
                 ckpt=args.ckpt, model=args.model, pretrained_path=args.pretrained_path,
                 device=args.device, train_area=args.train_area, steps=args.steps, cfg=args.cfg,
                 seed=args.seed, tag="gate_eval")
    if not args.skip_judge:
        run_judge(args, items)


if __name__ == "__main__":
    main()
