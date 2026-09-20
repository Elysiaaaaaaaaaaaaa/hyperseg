#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp/hyperseg
python_bin=/root/miniconda3/bin/python
baseline=runs/loveda_fewshot_valsplit/adapter_5shot_seed3407
output=runs/loveda_fewshot_valsplit/adapter_5shot_seed3407_mapped
"$python_bin" -c 'import torch; assert torch.cuda.is_available(), "GPU unavailable; enable GPU before launching"'
test -s "$baseline/selection.json"
test -s models/hyperseg_b3_best.pt
# Atomic directory creation prevents duplicate launches and overwriting a prior run.
mkdir "$output"
nohup env CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 "$python_bin" -u experiment/loveda_fewshot/train.py \
  --data-root /root/autodl-tmp/LoveDA --train-split Val --val-split Val \
  --init-checkpoint models/hyperseg_b3_best.pt --head-init mapped \
  --manifest "$baseline/selection.json" --output-dir "$output" \
  --shots 5 --domains urban rural --mode adapter --seed 3407 \
  --epochs 50 --steps-per-epoch 100 --crop-size 512 --eval-size 1024 \
  --batch-size 2 --eval-batch-size 1 --num-workers 4 --val-interval 5 \
  --lr 0.0001 --encoder-lr-multiplier 0.1 --weight-decay 0.01 --amp \
  > "$output.log" 2>&1 < /dev/null &
training_pid=$!
printf '%s\n' "$training_pid" > "$output.pid"
printf 'PID=%s\nLOG=%s/%s.log\n' "$training_pid" "$PWD" "$output"
