#!/usr/bin/env bash
# Launch the HyperSeg-UAV v2 resume-checkpoint export on server2 (RTX 4090).
#
# Usage: bash launch_server2.sh [window_size]
# The window size defaults to 768 but must be the value calibrated by
# v2_val_replay.py -- the v2 training script is not in the repository, so the
# resolution the checkpoint was selected at has to be measured, not assumed.
set -euo pipefail
size="${1:-768}"
project_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$project_root"
export PYTHONPATH="$project_root${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
task_dir="$project_root/runs/b3_resume_test2_20260930"
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
echo "Launching v2 export with window size $size"
nohup "$task_python" -u experiment/b3_resume_test2_20260930/infer_v2_test2.py \
    --size "$size" --overlap 0.5 \
    > "$task_dir/runner.log" 2>&1 < /dev/null &
task_pid=$!
printf '%s\n' "$task_pid" > "$task_dir/inference.pid"
printf 'Started inference PID=%s; log=%s/runner.log\n' "$task_pid" "$task_dir"
