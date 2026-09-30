#!/usr/bin/env bash
# Reproduce the checkpoint's recorded validation mIoU.
#
# val_sanity.py replays the exact training-time validation protocol (fixed
# runs/splits/val.txt, 768x768, batch 2, the same mean_iou) and compares it with the
# value stored inside the checkpoint. A mismatch means the Transformers-5 ->
# Transformers-4 key translation misaligned the backbone, so this is the strongest
# single check that the submitted weights load correctly.
#
# Expected: replayed 0.7420232149377345 vs recorded 0.7420226665631505 (diff 5.5e-7)
#
# Usage: scripts/03_eval.sh [data_root] [work_dir]
set -euo pipefail

cd "$(dirname "$0")/../hyperseg"
DATA="${1:-dataset/low_altitude_2026}"
WORK="${2:-experiment/round2_reproduce}"

export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

python experiment/b3_test2_20260929/val_sanity.py \
    --data "$DATA" \
    --split-dir "${WORK}/splits" \
    --checkpoint models/hyperseg_b3_best.pt

echo
echo "optional, still on the labelled split only:"
echo "  python tools/test_hyperseg.py --checkpoint models/hyperseg_b3_best.pt \\"
echo "      --data $DATA --split-dir ${WORK}/splits"
