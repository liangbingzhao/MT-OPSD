#!/bin/bash
# Gate worker for configs with gate_promote: true. Run it as a separate job next to training
# (it only reads/writes files under OUTPUT_DIR). Needs OPENAI_API_KEY for the GPT-4o judge.
#   OPENAI_API_KEY=... bash scripts/gate_worker.sh outputs/qwen_image_edit_2511
#   NUM_GPUS=2 bash scripts/gate_worker.sh outputs/qwen_image_edit_2511   # halves the eval time
set -e
RUN_DIR=${1:?usage: bash scripts/gate_worker.sh OUTPUT_DIR [extra args]}
shift
cd "$(dirname "$0")/.."
python gate_worker.py --run_dir "$RUN_DIR" --num_gpus "${NUM_GPUS:-1}" "$@"
