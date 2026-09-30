#!/usr/bin/env bash
# Check the export against the manual, then package it.
#
# Two levels:
#   1. tools/check_submission.py   (server-side, ships with the project)
#      names match the input, single-channel image, 1024x1024, values in 0..8
#   2. 01_预测结果/... check via 03_代码与复现/image-tools
#      the stricter block-level review used for this submission: parses the PNG
#      chunks so a palette-indexed image (PLTE/tRNS) cannot slip through, and diffs
#      the member list against the official test-set archive.
#
# Usage: scripts/05_check_submission.sh [data_root] [predictions_dir]
set -euo pipefail

cd "$(dirname "$0")/../hyperseg"
DATA="${1:-dataset/low_altitude_2026}"
PREDICTIONS="${2:-runs/b3_test2_reproduce/predictions}"

export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

python tools/check_submission.py "$PREDICTIONS" "$DATA/test_2/images"

echo
echo "block-level review: run experiment/b3_test2_20260929/check_manual_compliance.py"
echo "inside the development repository, or port the PNG-chunk part of it, which reads"
echo "struct/zipfile only. The recorded verdict for the submitted archive is in"
echo "05_实验证据/logs/manual_compliance_20260929.json (PASS: 1300 entries, colour"
echo "type 0 for every file, no PLTE/tRNS, names identical to the official test set 2)."
