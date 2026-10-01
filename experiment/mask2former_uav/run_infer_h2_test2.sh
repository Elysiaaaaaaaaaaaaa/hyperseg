#!/usr/bin/env bash
# Export H2 (Swin-L + Mask2Former, 20k) predictions on the 1300-image test_2 set.
#
#   nohup bash experiment/mask2former_uav/run_infer_h2_test2.sh \
#       >/dev/null 2>&1 < /dev/null &
#
# Protocol: MMSeg whole-image inference at native 1024x1024
# (`test_cfg mode='whole'`, `size_divisor=32`), the same channel H1 used for the
# 500-image preliminary export.  It is deliberately NOT the 512 px / 0.5 overlap
# sliding window used by the H3 / bgfix arms: MMSegmentation 1.2.2 slide inference
# updates the crop `img_shape` but leaves `pad_shape`, and Mask2FormerHead prefers
# `pad_shape`, so a slide export comes out wrongly sized (see README, "Evaluate and
# export").  Cross-protocol numbers vs bgfix must therefore be read as a protocol
# change, not a pure model change.
set -uo pipefail

project_root=/root/hyperseg
cd "$project_root"

runs="$project_root/runs/mask2former_uav"
images="$project_root/dataset/low_altitude_2026/test_2/images"
work="$runs/h2_swin_l_mask2former_20k_seed3407_fp32"
checkpoint="$work/best_mIoU_iter_20000.pth"
output="$runs/pred_test2_h2_20k"
python_bin="$project_root/.venv-mask2former/bin/python"
log="$runs/h2_test2_export.log"
expected=1300

export PYTHONPATH="$project_root/../mmsegmentation:$project_root${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

say() { printf '%s  %s\n' "$(date -Is)" "$*" >> "$log"; }

: > "$log"
say "H2 test_2 export start: $(ls "$images"/*.png 2>/dev/null | wc -l) images, whole-image 1024"
say "free before: $(df -h /root | tail -1 | awk '{print $4}')"

if [[ ! -f "$checkpoint" ]]; then
    say "ABORT: checkpoint missing at $checkpoint"
    exit 2
fi
say "checkpoint sha256: $(sha256sum "$checkpoint" | cut -c1-16)  size=$(du -h "$checkpoint" | cut -f1)"

rm -rf "$output"
mkdir -p "$output"
started=$(date +%s)
say "START export"

# run.py export runs MMSeg tools/test.py --out and then tools/check_submission.py,
# so a non-zero exit already means the masks failed the single-channel / 1024x1024 /
# 0..8 contract.
"$python_bin" -u experiment/mask2former_uav/run.py export \
    --model mask2former_swin_l \
    --checkpoint "$checkpoint" \
    --input "$images" \
    --output "$output" \
    --work-dir "$runs/h2_test2_export" \
    >> "$log" 2>&1
code=$?

written=$(ls "$output"/*.png 2>/dev/null | wc -l)
elapsed=$(( $(date +%s) - started ))
say "DONE export exit=$code png=$written elapsed=${elapsed}s"

if [[ "$code" -eq 0 && "$written" -eq "$expected" ]]; then
    tar -czf "$runs/pred_test2_h2_20k.tar.gz" -C "$runs" "pred_test2_h2_20k" \
        && say "packed pred_test2_h2_20k.tar.gz ($(du -h "$runs/pred_test2_h2_20k.tar.gz" | cut -f1))"
else
    say "WARNING export did not produce $expected predictions (exit=$code, got=$written); not packed"
fi
say "free after: $(df -h /root | tail -1 | awk '{print $4}')"
say "h2 test2 export driver done"
