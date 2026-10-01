#!/usr/bin/env bash
# Test any H2 (Swin-L + Mask2Former) checkpoint **without disturbing the running
# training job**: (1) labelled test.txt mIoU, (2) test_2 whole-image export.
#
#   bash run_h2_ckpt_test.sh <checkpoint> <label> [--skip-export]
#
# e.g.  bash run_h2_ckpt_test.sh \
#           /root/hyperseg/runs/mask2former_uav/h2_swin_l_mask2former_160k_seed3407_fp32/best_mIoU_iter_82000.pth \
#           iter82000
#
# Outputs (all under runs/mask2former_uav/):
#   h2_test_<label>.log                 full stdout of both steps
#   pred_test2_h2_<label>/             1300 masks
#   pred_test2_h2_<label>.tar.gz       packed archive (only when all 1300 pass)
#
# Both steps are deliberately light: 2 dataloader workers and ~2 GB of GPU memory,
# so they can share the card with a training run that is already holding ~9.7 GB.
set -uo pipefail

ckpt=${1:?usage: $0 <checkpoint> <label> [--skip-export]}
label=${2:?usage: $0 <checkpoint> <label> [--skip-export]}
skip_export=${3:-}

project_root=/root/hyperseg
cd "$project_root"

runs="$project_root/runs/mask2former_uav"
images="$project_root/dataset/low_altitude_2026/test_2/images"
python_bin="$project_root/.venv-mask2former/bin/python"
log="$runs/h2_test_${label}.log"
expected=1300

export PYTHONPATH="$project_root/../mmsegmentation:$project_root${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2

say() { printf '%s  %s\n' "$(date -Is)" "$*" >> "$log"; }

: > "$log"
say "=== ckpt test: label=$label ==="
say "checkpoint: $ckpt"
if [[ ! -f "$ckpt" ]]; then
    say "ABORT: checkpoint missing"
    exit 2
fi
say "ckpt size=$(du -h "$ckpt" | cut -f1) sha256=$(sha256sum "$ckpt" | cut -c1-16)"
say "free before: $(df -h /root | tail -1 | awk '{print $4}')  gpu: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader)"

# ---- step 1: labelled test.txt (699 images, independent of val) -------------
say "--- EVAL test.txt (699 labelled images) ---"
started=$(date +%s)
"$python_bin" -u experiment/mask2former_uav/run.py eval \
    --model mask2former_swin_l \
    --checkpoint "$ckpt" \
    --work-dir "$runs/h2_test_${label}_eval" \
    --num-workers 2 >> "$log" 2>&1
code=$?
say "eval exit=$code elapsed=$(( $(date +%s) - started ))s"
# MMSeg prints the metric block as per-class table + one trailing INFO line shaped
# "Iter(test) [699/699]  aAcc: .. mIoU: .. mAcc: .."; there is no "Summary:" header.
grep -E 'aAcc: [0-9.]+ +mIoU: [0-9.]+' "$log" | tail -1 \
    | sed -E 's/.*(aAcc: [0-9.]+ +mIoU: [0-9.]+ +mAcc: [0-9.]+).*/  summary: \1/' >> "$log"

# ---- step 2: test_2 whole-image export --------------------------------------
if [[ "$skip_export" != "--skip-export" ]]; then
    output="$runs/pred_test2_h2_${label}"
    say "--- EXPORT test_2 ($expected unlabelled, whole-image 1024) ---"
    rm -rf "$output"
    mkdir -p "$output"
    started=$(date +%s)
    "$python_bin" -u experiment/mask2former_uav/run.py export \
        --model mask2former_swin_l \
        --checkpoint "$ckpt" \
        --input "$images" \
        --output "$output" \
        --work-dir "$runs/h2_test_${label}_export" \
        --num-workers 2 >> "$log" 2>&1
    code=$?
    written=$(ls "$output"/*.png 2>/dev/null | wc -l)
    say "export exit=$code png=$written elapsed=$(( $(date +%s) - started ))s"
    if [[ "$code" -eq 0 && "$written" -eq "$expected" ]]; then
        tar -czf "$runs/pred_test2_h2_${label}.tar.gz" -C "$runs" "pred_test2_h2_${label}" \
            && say "packed pred_test2_h2_${label}.tar.gz $(du -h "$runs/pred_test2_h2_${label}.tar.gz" | cut -f1)"
    else
        say "WARNING: export incomplete (exit=$code got=$written); not packed"
    fi
fi

say "free after: $(df -h /root | tail -1 | awk '{print $4}')"
date -Is >> "$log" 2>/dev/null || true
say "=== ckpt test done ==="
