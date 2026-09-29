#!/usr/bin/env bash
# Run this script on server2 after GPU mode is enabled.
set -euo pipefail
project_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
cd "$project_root"
export PYTHONPATH="$project_root/../mmsegmentation:$project_root${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
task_dir="$project_root/runs/mask2former_uav/h3_test2_20260926"
task_python="$project_root/.venv-mask2former/bin/python"
"$task_python" -c 'import torch; assert torch.cuda.is_available(), "Enable GPU mode before launching"'
mkdir -p "$task_dir"
if [[ -f "$task_dir/inference.pid" ]] && kill -0 "$(cat "$task_dir/inference.pid")" 2>/dev/null; then
    echo 'Inference is already running.'
    exit 1
fi
if [[ -d "$task_dir/predictions" ]] && [[ -n "$(ls -A "$task_dir/predictions")" ]]; then
    echo 'Predictions already exist; inspect the previous run before launching again.'
    exit 1
fi
nohup "$task_python" -u experiment/mask2former_uav/test2_20260926/run_inference.py \
    > "$task_dir/runner.log" 2>&1 < /dev/null &
task_pid=$!
printf '%s\n' "$task_pid" > "$task_dir/inference.pid"
printf 'Started inference PID=%s; log=%s/runner.log\n' "$task_pid" "$task_dir"
