#!/bin/bash
# Multi-GPU MT-OPSD training (single node).
#   bash scripts/train.sh configs/qwen_image_edit_2511.yaml [extra --key value overrides]
#   NUM_GPUS=8 bash scripts/train.sh configs/firered_image_edit_1.0.yaml --output_dir outputs/fr
# max_steps / save_every / eval_every count GLOBAL samples, so changing NUM_GPUS changes the
# number of optimizer steps and the effective batch; the released recipes use 4 GPUs.
# Re-running the same command resumes from the newest checkpoint in output_dir.
set -e
CONFIG=${1:?usage: bash scripts/train.sh CONFIG [overrides...]}
shift
NUM_GPUS=${NUM_GPUS:-4}
PORT=${PORT:-29500}
cd "$(dirname "$0")/.."
accelerate launch --multi_gpu --num_machines 1 --num_processes "$NUM_GPUS" \
  --main_process_port "$PORT" --mixed_precision no --dynamo_backend no \
  train.py --config "$CONFIG" "$@"
