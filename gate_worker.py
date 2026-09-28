"""Gated teacher promotion: the sidecar job that runs next to training.

    python gate_worker.py --run_dir OUTPUT_DIR [--num_gpus 2]

It watches a training run's output dir, evaluates the newest checkpoint on the gate set
(mtopsd.gate.gate_eval: generation + GPT-4o judge) and writes gate/verdict-<step>.json.
Training (launched with gate_promote: true) polls those verdicts and hot-swaps its frozen
teacher to promoted checkpoints; training never waits for the gate. The two jobs only
share the filesystem. Before the first checkpoint the worker evaluates the base model,
which is the initial teacher.

Promotion rule, comparing the candidate with the current teacher at turn 10, then 8, then 5
(deltas in sessions out of 23; dSR = more successful sessions, dCol = fewer collapsed ones):
    dSR >= sr_bar and dCol >= -1                            -> promote (success-led)
    |dSR| <= 1 and dCol >= col_unlock                        -> promote (stability-led)
    -1 <= dSR <= sr_bar-1 and -1 <= dCol <= col_unlock-1     -> tie, look at the next turn
    otherwise                                                -> reject
All three levels tied -> reject. sr_bar / col_unlock and the first judged step come from the
run's config (gate_sr_bar / gate_col_unlock / gate_min_step). Restart-safe: state lives in
gate/worker_state.json.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import subprocess
import sys
import time

import yaml

from mtopsd.judge import aggregate

REPO = os.path.dirname(os.path.abspath(__file__))


def _atomic_json(path, obj):
    with open(path + ".tmp", "w") as f:
        json.dump(obj, f, indent=1)
    os.replace(path + ".tmp", path)


def _load(path):
    with open(path) as f:
        return json.load(f)


def checkpoint_steps(run_dir):
    out = []
    for d in glob.glob(os.path.join(run_dir, "checkpoint-*")):
        m = re.search(r"checkpoint-(\d+)$", d)
        if m and os.path.isfile(os.path.join(d, "training_state.pt")):
            out.append(int(m.group(1)))
    return sorted(out)


def cascade(cand, inc, sr_bar=2, col_unlock=5):
    """Returns (promote, per-level detail) for two gate reports on the same gate set."""
    n = cand["total"]
    if inc["total"] != n:
        raise RuntimeError(f"gate size mismatch: {n} vs {inc['total']}")
    cr, ir = cand["report"], inc["report"]
    levels = []
    for k in ("turn10", "turn8", "turn5"):
        d_sr = round(cr["success_rate"][k] * n) - round(ir["success_rate"][k] * n)
        d_col = round(ir["collapsed_rate"][k] * n) - round(cr["collapsed_rate"][k] * n)
        win_sr = d_sr >= sr_bar and d_col >= -1
        win_col = col_unlock > 0 and abs(d_sr) <= 1 and d_col >= col_unlock
        tie = (-1 <= d_sr <= sr_bar - 1) and (-1 <= d_col <= col_unlock - 1)
        levels.append({"level": k, "d_sr": d_sr, "d_col": d_col,
                       "path": "success-led" if win_sr else ("stability-led" if win_col else None)})
        if tie:
            continue
        return win_sr or win_col, levels
    return False, levels


def merge_shards(paths):
    per_sample = {}
    for p in paths:
        per_sample.update(_load(p)["per_sample"])
    n_turn = max(int(k[4:]) for r in per_sample.values() for k in r if k.startswith("turn"))
    return {"total": len(per_sample), "done": len(per_sample),
            "report": aggregate(per_sample, n_turn), "per_sample": per_sample}


def evaluate(args, cfg, name, ckpt):
    """Gate eval of one checkpoint (or the base model with ckpt=None), sharded over GPUs."""
    out = os.path.join(args.run_dir, "gate", "evals", name)
    base_cmd = [sys.executable, "-m", "mtopsd.gate.gate_eval",
                "--train_area", str(cfg["train_area"]), "--steps", str(cfg["rollout_steps"]),
                "--cfg", str(cfg["rollout_cfg"]), "--api_model", args.api_model]
    base_cmd += ["--ckpt", ckpt] if ckpt else ["--model", cfg["model"]]
    if cfg.get("pretrained_path"):
        base_cmd += ["--pretrained_path", cfg["pretrained_path"]]
    if args.limit:
        base_cmd += ["--limit", str(args.limit)]
    n = args.num_gpus
    procs = []
    for i in range(n):
        o = out if n == 1 else f"{out}_s{i}"
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(i)) if n > 1 else None
        procs.append(subprocess.Popen(base_cmd + ["--out", o, "--shard", str(i),
                                                  "--num_shards", str(n)], cwd=REPO, env=env))
    rcs = [p.wait() for p in procs]
    if any(rcs):
        raise RuntimeError(f"gate eval of {name} failed: exit codes {rcs}")
    if n == 1:
        return os.path.join(out, "judge", "results.json")
    merged = merge_shards([os.path.join(f"{out}_s{i}", "judge", "results.json") for i in range(n)])
    os.makedirs(os.path.join(out, "judge"), exist_ok=True)
    path = os.path.join(out, "judge", "results.json")
    _atomic_json(path, merged)
    return path


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run_dir", required=True, help="the training run's output_dir.")
    ap.add_argument("--num_gpus", type=int, default=1,
                    help="gate sessions are split over this many GPUs (one process each).")
    ap.add_argument("--until_step", type=int, default=0,
                    help="exit after judging a checkpoint >= this step (default: max_steps "
                         "rounded down to the save grid).")
    ap.add_argument("--min_step", type=int, default=None,
                    help="ignore earlier checkpoints (default: the run's gate_min_step).")
    ap.add_argument("--poll_secs", type=int, default=120)
    ap.add_argument("--idle_hours", type=float, default=3.0,
                    help="exit when no new checkpoint appears for this long.")
    ap.add_argument("--api_model", default="gpt-4o")
    ap.add_argument("--limit", type=int, default=0, help="smoke test: N sessions per GPU.")
    args = ap.parse_args()

    cfg_path = os.path.join(args.run_dir, "config_resolved.yaml")
    while not os.path.isfile(cfg_path):          # the training job writes it at startup
        print(f"[gate] waiting for {cfg_path}", flush=True)
        time.sleep(args.poll_secs)
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    if not args.until_step:
        world = cfg.get("world", 1)
        grid = max(1, cfg["save_every"] // world) * world
        args.until_step = (cfg["max_steps"] // world * world) // grid * grid
    if args.min_step is None:
        args.min_step = cfg.get("gate_min_step", 0)

    gate_dir = os.path.join(args.run_dir, "gate")
    os.makedirs(gate_dir, exist_ok=True)
    state_path = os.path.join(gate_dir, "worker_state.json")
    if os.path.isfile(state_path):
        st = _load(state_path)
    else:
        print("[gate] evaluating the base model (initial teacher)", flush=True)
        st = {"incumbent": {"name": "base", "step": -1,
                            "results": evaluate(args, cfg, "base", None)},
              "done": []}
        _atomic_json(state_path, st)
    print(f"[gate] teacher={st['incumbent']['name']} judged={st['done']} "
          f"from={args.min_step} until={args.until_step} sr_bar={cfg['gate_sr_bar']} "
          f"col_unlock={cfg['gate_col_unlock']}",
          flush=True)

    last_new = time.time()
    while True:
        done = set(st["done"])
        todo = [s for s in checkpoint_steps(args.run_dir) if s >= args.min_step and s not in done]
        if not todo:
            if done and max(done) >= args.until_step:
                print("[gate] reached the last checkpoint -> done", flush=True)
                return
            if time.time() - last_new > args.idle_hours * 3600:
                print(f"[gate] no new checkpoint for {args.idle_hours} h -> exit", flush=True)
                return
            time.sleep(args.poll_secs)
            continue
        last_new = time.time()
        step = max(todo)                           # newest only; older ones are skipped
        results = evaluate(args, cfg, f"step_{step}",
                           os.path.join(args.run_dir, f"checkpoint-{step}"))
        cand, inc = _load(results), _load(st["incumbent"]["results"])
        promote, levels = cascade(cand, inc, cfg["gate_sr_bar"], cfg["gate_col_unlock"])
        _atomic_json(os.path.join(gate_dir, f"verdict-{step}.json"), {
            "step": step, "promote": promote, "cascade": levels,
            "teacher": st["incumbent"]["name"], "candidate_results": results,
            "skipped": [s for s in todo if s != step], "time": time.strftime("%Y-%m-%d %H:%M:%S")})
        print(f"[gate] step {step}: promote={promote} "
              f"{[(l['level'], l['d_sr'], l['d_col']) for l in levels]}", flush=True)
        if promote:
            st["incumbent"] = {"name": f"step_{step}", "step": step, "results": results}
        st["done"] = sorted(done | set(todo))
        _atomic_json(state_path, st)
        if step >= args.until_step:
            print("[gate] judged the last checkpoint -> done", flush=True)
            return


if __name__ == "__main__":
    main()
