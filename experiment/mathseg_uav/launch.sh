#!/usr/bin/env bash
# Usage: PYTHON_BIN=/path/to/python bash experiment/mathseg_uav/launch.sh OUTPUT_ROOT [run.py options] -- [train.py options]
set -euo pipefail
if [[ $# -lt 1 ]]; then
    echo "Usage: $0 OUTPUT_ROOT [queue options] -- [training options]" >&2
    exit 2
fi
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
run_root="$1"
shift
mkdir -p -- "$run_root"
run_root="$(cd -- "$run_root" && pwd)"
python_bin="${PYTHON_BIN:-python3}"
nohup "$python_bin" -u "$script_dir/run.py" run --output-root "$run_root" "$@" \
    >> "$run_root/queue.log" 2>&1 < /dev/null &
queue_pid=$!
echo "$queue_pid" > "$run_root/queue.pid"
echo "Queue PID: $queue_pid; log: $run_root/queue.log"
