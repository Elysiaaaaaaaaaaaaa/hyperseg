#!/usr/bin/env bash
# Background launcher for the H3 background-collapse fix.
#
# Usage (on the server, from the repository root):
#   bash experiment/h3_bgfix_20260930/launch_h3_bgfix.sh full
#   bash experiment/h3_bgfix_20260930/launch_h3_bgfix.sh loss
#   bash experiment/h3_bgfix_20260930/launch_h3_bgfix.sh full full_20k --max-iters 20000 --val-interval 1000
#
# $1 = preset (baseline | loss | aug | full), $2 = run tag.  Anything after that is
# forwarded verbatim to the trainer, so a screening budget or an ablation override does
# not need a code change.
#
# Refuses to start if the same run is already live or if its work directory already holds
# a finished run, so an accidental double launch cannot clobber a checkpoint.
set -euo pipefail

preset="${1:-full}"
tag="${2:-${preset}}"
if [[ $# -ge 2 ]]; then
    shift 2
elif [[ $# -ge 1 ]]; then
    shift 1
fi

project_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$project_root"

export PYTHONPATH="$project_root/../mmsegmentation:$project_root${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

work_dir="$project_root/runs/mask2former_uav/h3_bgfix_${tag}_seed3407"
mkdir -p "$work_dir"

exec 9>"$work_dir/train.lock"
flock -n 9 || { echo "another H3 bgfix run holds $work_dir/train.lock"; exit 1; }
if [[ -f "$work_dir/config.json" ]]; then
    echo "work dir already contains a run: $work_dir"
    echo "inspect it, then remove it or resume explicitly with --resume"
    exit 1
fi

trap 'result=$?; if [[ $result -ne 0 ]]; then printf "%s\n" "$result" > "$work_dir/launcher_exit_code"; fi' EXIT

python_bin="$project_root/.venv-mask2former/bin/python"
backbone="$project_root/models/swin_l/swin_large_patch4_window12_384_22k_20220412-6580f57d.pth"

{
    echo "preset=$preset tag=$tag"
    date -Is
    "$python_bin" - <<'PY'
import torch
print(f"torch {torch.__version__} cuda_available={torch.cuda.is_available()} "
      f"device_count={torch.cuda.device_count()}")
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
PY
    nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv || true
    sha256sum "$backbone" || echo "WARNING: backbone checkpoint not found at $backbone"
} 2>&1 | tee "$work_dir/preflight.log"

if ! "$python_bin" -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)"; then
    echo "CUDA is not available; refusing to start a training run" | tee -a "$work_dir/preflight.log"
    exit 2
fi

if [[ ! -d "$project_root/../mmsegmentation" ]]; then
    echo "FATAL: sibling MMSegmentation checkout missing at $project_root/../mmsegmentation" | tee -a "$work_dir/preflight.log"
    exit 2
fi

smoke_status=0
"$python_bin" -u experiment/h3_bgfix_20260930/smoke_test.py 2>&1 | tee "$work_dir/smoke_test.log" || smoke_status=$?
if [[ $smoke_status -eq 0 ]] && grep -q "ALL CHECKS PASSED" "$work_dir/smoke_test.log"; then
    echo "smoke test passed" | tee -a "$work_dir/preflight.log"
else
    echo "FATAL: smoke test failed (exit=$smoke_status); not starting training" \
        | tee -a "$work_dir/preflight.log"
    echo "see $work_dir/smoke_test.log for the failing checks" | tee -a "$work_dir/preflight.log"
    exit 3
fi

echo "starting training" | tee -a "$work_dir/preflight.log"

# The training run has to be detached, but its exit status still matters -- a launcher that
# backgrounds the job and then exits cannot report it.  So generate a tiny wrapper that
# runs the trainer and writes the real exit code, then detach the wrapper.
wrapper="$work_dir/run_train.sh"
{
    printf '#!/usr/bin/env bash\n'
    printf 'cd %q\n' "$project_root"
    printf 'export PYTHONPATH=%q\n' "$PYTHONPATH"
    printf 'export OMP_NUM_THREADS=4\n'
    printf 'export MKL_NUM_THREADS=4\n'
    printf '%q -u experiment/h3_bgfix_20260930/train_h3_bgfix.py' "$python_bin"
    printf ' --preset %q' "$preset"
    printf ' --data-root %q' "$project_root/dataset/low_altitude_2026"
    printf ' --split-dir %q' "$project_root/runs/splits"
    printf ' --work-dir %q' "$work_dir"
    printf ' --backbone-checkpoint %q' "$backbone"
    printf ' --batch-size 2 --accumulative-counts 1 --size 512'
    printf ' --max-iters 160000 --val-interval 2800 --num-workers 4'
    printf ' --seed 3407 --with-checkpointing'
    for extra in "$@"; do printf ' %q' "$extra"; done
    printf '\n'
    printf 'status=$?\n'
    printf 'printf "%%s\\n" "$status" > %q\n' "$work_dir/exit_code"
    printf 'date -Is >> %q\n' "$work_dir/exit_code"
} > "$wrapper"
chmod +x "$wrapper"
{
    echo "--- wrapper ---"
    cat "$wrapper"
    echo "---------------"
} >> "$work_dir/preflight.log"

nohup "$wrapper" >> "$work_dir/train.log" 2>&1 < /dev/null &
train_pid=$!
printf '%s\n' "$train_pid" > "$work_dir/train.pid"
echo "pid=$train_pid work_dir=$work_dir"
echo "follow with: tail -f $work_dir/train.log"
echo "the run is finished when $work_dir/exit_code appears (0 = success)"
