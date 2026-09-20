#!/usr/bin/env bash
# Invoke with nohup; stdout/stderr should be redirected to the work directory.
set -euo pipefail
cd /root/autodl-tmp/hyperseg
export PYTHONPATH=/root/autodl-tmp/mmsegmentation:/root/autodl-tmp/hyperseg
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
work_dir=/root/autodl-tmp/work_dirs/mask2former_uav/h3_swin_l_hyperseg_160k_seed3407_fp32
mkdir -p "$work_dir"
exec 9>"$work_dir/train.lock"
flock -n 9 || { echo 'H3 is already running'; exit 1; }
if [[ -f "$work_dir/config.json" ]]; then
    echo 'Work directory already contains a run; use explicit --resume after inspection.'
    exit 1
fi
trap 'result=$?; printf "%s\n" "$result" > "$work_dir/exit_code"' EXIT
date -Is
sha256sum models/swin_l/swin_large_patch4_window12_384_22k_20220412-6580f57d.pth
.venv-mask2former/bin/python -u experiment/mask2former_uav/train_h3_hyperseg.py \
    --data-root /root/autodl-tmp/data/low_altitude_2026 \
    --split-dir /root/autodl-tmp/hyperseg/runs/splits \
    --work-dir "$work_dir" \
    --backbone-checkpoint /root/autodl-tmp/hyperseg/models/swin_l/swin_large_patch4_window12_384_22k_20220412-6580f57d.pth \
    --batch-size 2 --accumulative-counts 1 --size 512 \
    --max-iters 160000 --val-interval 2800 --num-workers 4 \
    --seed 3407 --with-checkpointing
date -Is
