#!/usr/bin/env bash
# Retrain HyperSeg-UAV / MiT-B3 from the fixed split.
#
# The submitted checkpoint was selected by validation mIoU, so the same selection
# rule applies here. See configs/train_config.json for the defaults and for what is
# and is not recoverable about the original run.
#
# Usage: scripts/02_train.sh [work_dir]
set -euo pipefail

cd "$(dirname "$0")/../hyperseg"
WORK="${1:-experiment/round2_reproduce}"
OUT="${OUT:-runs/round2_reproduce/hyperseg_b3_best.pt}"

export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

mkdir -p "$(dirname "$OUT")"
python tools/train_hyperseg.py \
    --data "$WORK/train" \
    --split-dir "$WORK/splits" \
    --out "$OUT" \
    --size 768 \
    --batch-size 2 \
    --lr 1e-4 \
    --encoder-lr-multiplier 0.1 \
    --seed 3407 \
    --no-amp \
    --epochs "${EPOCHS:-60}"

echo "checkpoint: $OUT"
