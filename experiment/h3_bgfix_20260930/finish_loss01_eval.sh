#!/usr/bin/env bash
# GPU 恢复后的收口脚本：把 `loss01`（false_bg_weight=0.1）补齐到与其它四臂同口径。
#
#   nohup bash experiment/h3_bgfix_20260930/finish_loss01_eval.sh \
#       >/dev/null 2>&1 < /dev/null &
#
# 两个阶段都必须独占 GPU（单进程已把卡吃到 96%，叠加只会两边都变慢），所以串行：
#
#   1. test_2 无标注推理 —— 512 px 窗口 / 0.5 overlap，与已发布的
#      `swin_l_hyperseg_复赛.zip` 协议一致，因此可直接对比「背景塌陷是否被削」。
#   2. 五个 best.pt 的同口径代理评估（有标注 val × 0.5/0.625/0.75 三尺度），
#      补上 `loss01` 这一列，同时刷新 `proxy_20k_all_arms.json`。
#
# 与 `run_infer_test2_then_loss01.sh` 的区别：那个会在末尾自动起训练，这个只做评估，
# 且**不截断**共享日志（用 `>>`），以免覆盖当晚已有的记录。
set -uo pipefail

project_root=/root/hyperseg
cd "$project_root"

runs="$project_root/runs/mask2former_uav"
images="$project_root/dataset/low_altitude_2026/test_2/images"
python_bin="$project_root/.venv-mask2former/bin/python"
log="$runs/h3_bgfix_finish_loss01.log"

export PYTHONPATH="$project_root/../mmsegmentation:$project_root${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

say() { printf '%s  %s\n' "$(date -Is)" "$*" >> "$log"; }

say "=== finish_loss01 start ==="

if ! "$python_bin" -c 'import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)'; then
    say "ABORT: CUDA 不可用（/dev/nvidia0 多半又掉了），先在控制台重启实例"
    exit 1
fi

expected=1300
arm=loss01
checkpoint="$runs/h3_bgfix_${arm}_20k_seed3407/best.pt"
output="$runs/pred_test2_${arm}_20k"

# ---------------- 阶段 1：test_2 推理 ----------------
if [[ ! -f "$checkpoint" ]]; then
    say "ABORT: 缺检查点 $checkpoint"
    exit 1
fi

rm -rf "$output"
mkdir -p "$output"
started=$(date +%s)
say "START infer $arm -> $output"

"$python_bin" -u experiment/mask2former_uav/infer_h3_hyperseg.py \
    --checkpoint "$checkpoint" \
    --input "$images" \
    --output "$output" \
    --size 512 --overlap 0.5 >> "$log" 2>&1
code=$?

written=$(ls "$output"/*.png 2>/dev/null | wc -l)
elapsed=$(( $(date +%s) - started ))
say "DONE infer $arm exit=$code png=$written elapsed=${elapsed}s"

if [[ "$code" -eq 0 && "$written" -eq "$expected" ]]; then
    tar -czf "$runs/pred_test2_${arm}_20k.tar.gz" -C "$runs" "pred_test2_${arm}_20k" \
        && say "packed pred_test2_${arm}_20k.tar.gz ($(du -h "$runs/pred_test2_${arm}_20k.tar.gz" | cut -f1))"
else
    say "WARNING: $arm 只产出 $written/$expected 张（exit=$code），未打包"
fi

# ---------------- 阶段 2：五臂代理评估 ----------------
say "START eval_proxy (5 arms)"

eval_cmd=("$python_bin" -u experiment/h3_bgfix_20260930/eval_proxy.py)
for a in baseline loss aug full loss01; do
    eval_cmd+=(--checkpoint "runs/mask2former_uav/h3_bgfix_${a}_20k_seed3407/best.pt")
done
eval_cmd+=(--output experiment/h3_bgfix_20260930/results/proxy_20k_all_arms.json)

"${eval_cmd[@]}" >> "$log" 2>&1
say "DONE eval_proxy exit=$?"

say "free at end: $(df -h /root | tail -1 | awk '{print $4}')"
say "=== finish_loss01 done ==="
