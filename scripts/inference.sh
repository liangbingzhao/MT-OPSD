#!/bin/bash
# Ten-turn editing example. CKPT = an MT-OPSD checkpoint dir (e.g. downloaded from HF).
set -e
CKPT=${1:?usage: bash scripts/inference.sh CKPT_DIR [IMAGE] [OUTPUT_DIR]}
IMAGE=${2:-assets/gate_set/images/gate01.png}
OUT=${3:-outputs/inference_example}
cd "$(dirname "$0")/.."
python inference.py --ckpt "$CKPT" --image "$IMAGE" --output_dir "$OUT" --instructions \
  "Repaint the whole scene as a textured oil painting with visible brushstrokes." \
  "Change the cat's blue collar to bright red." \
  "Shift the whole image to a warm amber color temperature." \
  "Add a small bird perched outside on the window ledge." \
  "Increase the overall contrast for a punchy look." \
  "Make the cat stand upright on its four legs instead of sitting." \
  "Convert the entire image to black and white, full grayscale with no color." \
  "Replace the potted cactus with a small fern in the same pot." \
  "Add a second, smaller cat sitting beside the first cat." \
  "Remove the bird from the window ledge."
