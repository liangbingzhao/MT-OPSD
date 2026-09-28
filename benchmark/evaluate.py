"""Score generated LME-Bench sessions with the GPT-4o judge: success rate (SR) and collapse
rate (CR) at every turn.

    export OPENAI_API_KEY=...
    python benchmark/evaluate.py --samples outputs/lme/mtopsd_qwen [--samples outputs/lme/qwen_base ...]

<samples>/<id>/ must hold source.png and turn1.png ... turn10.png (see benchmark/sample.py;
any editor can be evaluated by writing its outputs in this layout). Results go to
<samples>/judge/{results.json, summary.csv}. Both passes cache every call, so an interrupted
run resumes without paying twice. One model = about 2,000 GPT-4o calls.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from mtopsd import judge  # noqa: E402

REPORT_TURNS = (3, 5, 8, 10)


def _atomic_json(obj, path):
    with open(path + ".tmp", "w") as f:
        json.dump(obj, f, indent=1)
    os.replace(path + ".tmp", path)


def evaluate(samples, items, args):
    client = (os.environ["OPENAI_API_KEY"], args.api_model)
    n_turn = len(items[0]["turns"])
    res_dir = os.path.join(samples, "judge")
    cache1 = os.path.join(res_dir, "pass1")
    cache2_path = os.path.join(res_dir, "pass2.json")
    os.makedirs(cache1, exist_ok=True)

    sessions = []
    for it in items:
        d = os.path.join(samples, it["id"])
        paths = [os.path.join(d, "source.png")] + [os.path.join(d, f"turn{t}.png")
                                                   for t in range(1, n_turn + 1)]
        missing = [p for p in paths if not os.path.isfile(p)]
        if missing:
            raise SystemExit(f"[lme] {samples}: {it['id']} is incomplete (missing {missing[0]})")
        sessions.append((it["id"], it["turns"], paths))

    # pass 1: prompt following / consistency (-> SR), cached per session
    pass1 = {}
    for sid, _, _ in sessions:
        p = os.path.join(cache1, f"{sid}.json")
        if os.path.isfile(p):
            with open(p) as f:
                pass1[sid] = json.load(f)

    def one1(s):
        sid, turns, paths = s
        r = judge.score_session(client, turns, paths)
        _atomic_json(r, os.path.join(cache1, f"{sid}.json"))
        return sid, r

    todo1 = [s for s in sessions if s[0] not in pass1 or "error" in pass1[s[0]]]
    print(f"[lme] {samples}: pass 1 on {len(todo1)} sessions ({len(sessions) - len(todo1)} cached)",
          flush=True)
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for sid, r in ex.map(one1, todo1):
            pass1[sid] = r

    # pass 2: content loss / surface corruption (-> CR), cached per (session, turn)
    pass2 = {}
    if os.path.isfile(cache2_path):
        with open(cache2_path) as f:
            pass2 = json.load(f)
    todo2 = [(sid, k, paths[0], paths[k], turns[:k]) for sid, turns, paths in sessions
             for k in range(1, n_turn + 1) if str(k) not in pass2.get(sid, {})]
    print(f"[lme] {samples}: pass 2 on {len(todo2)} frames", flush=True)

    def one2(j):
        sid, k, src, cur, ins = j
        return sid, k, judge.score_collapse_turn(client, src, cur, ins)

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for i, (sid, k, v) in enumerate(ex.map(one2, todo2), 1):
            if v is not None:
                pass2.setdefault(sid, {})[str(k)] = v
            if i % 50 == 0:
                _atomic_json(pass2, cache2_path)
    _atomic_json(pass2, cache2_path)

    failed1 = [sid for sid, r in pass1.items() if "error" in r]
    failed2 = sum(1 for sid, _, _ in sessions for k in range(1, n_turn + 1)
                  if str(k) not in pass2.get(sid, {}))
    if failed1 or failed2:
        # a judge outage must not look like a bad model; rerun to retry only these calls
        raise SystemExit(f"[lme] {samples}: {len(failed1)} sessions / {failed2} frames were not "
                         "scored (API errors); rerun the same command to retry them")

    scores = {sid: {int(t): e for t, e in d.items()} for sid, d in pass2.items()}
    rep = judge.lme_report(pass1, scores, n_turn, len(items))
    _atomic_json({"api_model": args.api_model, "n_sessions": len(items), "report": rep,
                  "pass1": pass1, "pass2": pass2}, os.path.join(res_dir, "results.json"))
    with open(os.path.join(res_dir, "summary.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["turn", "SR", "CR"])
        for t in range(1, n_turn + 1):
            w.writerow([t, round(rep["success_rate"][f"turn{t}"], 4),
                        round(rep["collapse_rate"][f"turn{t}"], 4)])
    return rep


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--samples", action="append", required=True, help="repeatable.")
    ap.add_argument("--sessions", default=os.path.join(HERE, "lme_bench.json"))
    ap.add_argument("--api_model", default="gpt-4o")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--limit", type=int, default=0, help="first N sessions only (smoke tests).")
    args = ap.parse_args()
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("[lme] set OPENAI_API_KEY")
    with open(args.sessions) as f:
        items = json.load(f)["items"]
    if args.limit:
        items = items[: args.limit]

    rows = []
    for s in args.samples:
        rep = evaluate(os.path.abspath(s), items, args)
        rows.append((os.path.basename(os.path.normpath(s)),
                     [rep["success_rate"][f"turn{t}"] for t in REPORT_TURNS],
                     [rep["collapse_rate"][f"turn{t}"] for t in REPORT_TURNS]))
    head = " ".join(f"T{t:<4}" for t in REPORT_TURNS)
    print(f"\n{'model':<32} SR {head}  CR {head}")
    for name, sr, cr in rows:
        print(f"{name[:32]:<32}    " + " ".join(f"{x:.2f} " for x in sr)
              + "     " + " ".join(f"{x:.2f} " for x in cr))


if __name__ == "__main__":
    main()
