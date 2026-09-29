#!/usr/bin/env bash
set -u

root=/root/autodl-tmp/work_dirs/mathseg_fewshot_20260922
common=(
  --data-root /root/autodl-tmp/LoveDA
  --manifest-dir /root/autodl-tmp/mathseg_fewshot_20260922/runs/loveda_manual/export_20260912_101105_509451
  --init-checkpoint /root/autodl-tmp/work_dirs/mathseg_uav/m3_seed3407/best.pt
  --modes semantic-head --steps 2000 --num-workers 4 --amp --device cuda
  --model mathseg --label-policy standard --checkpoint-format compact
)

for shots in 2 5 10; do
  /root/miniconda3/bin/python -u \
    /root/autodl-tmp/mathseg_fewshot_20260922/experiment/mathseg_uav/fewshot_loveda.py \
    "${common[@]}" --shots "$shots" --seeds 3407 \
    --output-root "$root/loveda/m3/semantic_head_2000/$shots" \
    >> "$root/m3_loveda_resume.log" 2>&1 || exit $?
done
