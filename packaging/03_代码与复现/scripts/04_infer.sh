#!/usr/bin/env bash
# Export predictions for the round-two test set.
#
# Protocol: FP32, 768x768 sliding window, 0.5 overlap, no TTA. 768 matches the
# resolution the checkpoint was trained and selected at; TTA is deliberately off
# because the submitted archive was produced without it.
#
# Usage: scripts/04_infer.sh [data_root] [output_dir]
set -euo pipefail

cd "$(dirname "$0")/../hyperseg"
DATA="${1:-dataset/low_altitude_2026}"
OUTPUT="${2:-runs/b3_test2_reproduce/predictions}"

# `python tools/infer_hyperseg.py` puts only `tools/` on sys.path, not the project
# root, so `import hyperseg_uav` needs the root on PYTHONPATH. (infer_hyperseg.py
# also bootstraps sys.path itself; this keeps the scripts uniform.)
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

# torch spawns one thread per visible core. On hosts whose cgroup quota is far
# below that count this makes the run many times slower, so cap it.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"

mkdir -p "$OUTPUT"
python tools/infer_hyperseg.py \
    --checkpoint models/hyperseg_b3_best.pt \
    --input "$DATA/test_2/images" \
    --output "$OUTPUT" \
    --size 768 \
    --overlap 0.5

echo "predictions: $OUTPUT ($(find "$OUTPUT" -name '*.png' | wc -l) files)"
