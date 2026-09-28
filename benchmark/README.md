# LME-Bench

**L**ong-horizon **M**ulti-turn image **E**diting benchmark: 100 ten-turn editing sessions.
Existing multi-turn benchmarks stop at five turns; LME-Bench is built to expose what happens
after that, when every edit is applied to the model's own previous output.

- 100 source images (1024×1024, generated with Z-Image-Turbo), 10 categories × 10 sessions:
  pet, wild, person, food, vehicle, interior, nature, object, plant, urban.
- Each session has 10 instructions, authored against its source image, mixing 6 local edits
  (add / remove / count / color / material / size / replace / state) with 4 global edits:
  hard globals (art style, black-and-white or sepia grade, weather / time of day) and soft
  globals (color temperature, brightness / contrast). Hard globals come early, so later turns
  edit an already transformed image.
- Instructions only move forward (no "undo" or "restore"): the editor only ever sees the
  previous turn's output.

`lme_bench.json` lists the sessions: `id`, `category`, `source_prompt` (the prompt the
source image was generated from), `turns` and per-turn `labels` (`hard` / `soft` / `local`).

## Download

```bash
hf download metazlb/LME-Bench --repo-type dataset --local-dir benchmark/data
# -> benchmark/data/images/0001.png ... 0100.png, plus lme_bench.json and metadata.jsonl
#    (the same session list; metadata.jsonl feeds the dataset viewer on the Hub)
```

## 1. Generate the sessions

```bash
# an MT-OPSD checkpoint (backbone and sampling settings come from its adapter_config.json)
python benchmark/sample.py --ckpt /path/to/mtopsd-qwen-image-edit-2511 --out outputs/lme/mtopsd_qwen

# base models, with the protocol used in the paper
python benchmark/sample.py --model qwen-image-edit-2511   --out outputs/lme/qwen_base
python benchmark/sample.py --model firered-image-edit-1.0 --out outputs/lme/firered_base
python benchmark/sample.py --model flux2-klein-base-9b --steps 30 --out outputs/lme/flux2_klein_base

# 4 GPUs: one process per GPU, same --out
for i in 0 1 2 3; do CUDA_VISIBLE_DEVICES=$i python benchmark/sample.py --ckpt CKPT --out OUT --shard $i --num_shards 4 & done; wait
```

Protocol: each turn edits the previous turn's output; 512×512 working area (aspect
preserved), 30 sampling steps, true CFG 4.0, the same seed (42) at every turn. One session
takes about a minute on an H100 for the Qwen family.

To evaluate any other editor, write its outputs as `<out>/<id>/source.png` (the given source
image) and `<out>/<id>/turn1.png ... turn10.png`, where turn *k* is the result of
instruction *k* applied to turn *k*−1.

## 2. Judge

```bash
export OPENAI_API_KEY=...        # an OpenRouter key (sk-or-...) also works
python benchmark/evaluate.py --samples outputs/lme/mtopsd_qwen --samples outputs/lme/qwen_base
```

This prints SR and CR at turns 3 / 5 / 8 / 10 and writes
`<samples>/judge/{summary.csv, results.json}` with every turn and every raw judgment. GPT-4o
is called about 2,000 times per model (about $9); all calls are cached, so rerunning after an
interruption only makes the missing ones.

## Metrics

Both come from a GPT-4o judge (`mtopsd/judge.py`) that runs two passes per session.

- **Success rate SR@k**: the fraction of sessions in which every turn 1..k succeeded. Pass 1
  looks at the whole session up to turn *t* and scores turn *t*: `prompt_following` (did the
  requested change happen, judged as the change from turn *t*−1) and `consistency` (is
  everything that should be preserved still there; for global edits, the subjects and
  composition must survive the transform). Turn *t* succeeds iff both scores are > 6. As in
  MSE-Bench, a session stops being scored for success at its first failed turn.
- **Collapse rate CR@k**: the fraction of sessions that have collapsed by turn *k*. Pass 2
  shows the source and the turn-*t* image, lists the instructions applied so far, and asks
  three factual questions: `content_loss` (0 all main objects identifiable / 1 one is gone /
  2 no longer the scene), `surface_corruption` (0 none / 1 visible but survivable /
  2 dominates the frame) and `appearance_violation` (recorded only). A frame is destroyed iff
  content_loss + surface_corruption ≥ 2, and a session collapses at the first of two
  consecutive destroyed turns and stays collapsed. Requested styles (pixel art, oil
  painting, grayscale) are explicitly not degradation.

## Reference results

From the paper (same protocol and judge):

| Model | SR@3 | SR@5 | SR@8 | SR@10 | CR@3 | CR@5 | CR@8 | CR@10 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen-Image-Edit-2511 | 0.98 | 0.43 | 0.08 | 0.03 | 0.00 | 0.10 | 0.45 | 0.55 |
| + MT-OPSD | 1.00 | 0.91 | 0.63 | 0.44 | 0.00 | 0.01 | 0.02 | 0.02 |
| FireRed-Image-Edit-1.0 | 0.95 | 0.57 | 0.28 | 0.15 | 0.01 | 0.16 | 0.51 | 0.61 |
| + MT-OPSD | 0.98 | 0.88 | 0.64 | 0.52 | 0.00 | 0.00 | 0.01 | 0.03 |
| FLUX.2 [klein] base 9B (30 steps) | 0.92 | 0.52 | 0.29 | 0.12 | 0.00 | 0.01 | 0.17 | 0.25 |
| + MT-OPSD | 0.95 | 0.76 | 0.57 | 0.38 | 0.00 | 0.00 | 0.02 | 0.04 |

With 100 sessions one session is 0.01, and the judge adds noise of a few points at deep turns,
so differences of a few sessions are within noise.
