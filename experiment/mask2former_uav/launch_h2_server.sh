#!/usr/bin/env bash
# H2: Swin-L + Mask2Former -- the 20k competition-baseline arm and its long-budget arms.
#
#   bash launch_h2_server.sh smoke      # 20-iter gate only, then exit
#   bash launch_h2_server.sh full       # smoke gate, then the real 20k run (nohup this)
#   bash launch_h2_server.sh long       # smoke gate, then a from-scratch run of H2L_MAX_ITERS
#   bash launch_h2_server.sh resume60k  # continue the finished 20k run to 60k (nohup this)
#
# Env knobs:
#   H2_VAL_INTERVAL=1000    validation / checkpoint cadence of the `full` arm
#   H2_MAX_ITERS=20000      training budget of the `full` arm
#   H2L_MAX_ITERS=160000    training budget of the `long` arm
#   H2L_VAL_INTERVAL=2000   validation cadence of the `long` arm (2000 hits iter 20000 exactly)
#   H2L_RESUME=1            restart the `long` arm from its own last_checkpoint (crash recovery)
#   H2_WITH_CP=1            Swin gradient checkpointing (OOM fallback)
#   H2B_TOTAL_ITERS=60000   budget of the `resume60k` arm
#   H2B_RESUMED_ITER=20000  iteration the source checkpoint stopped at
#   H2B_SRC_CKPT=...        override the source checkpoint of `resume60k`
#
# `long` vs `resume60k` -- not the same experiment
# ------------------------------------------------
# `long` runs from scratch with the PolyLR annealed over the whole budget, so its
# trajectory *differs* from the 20k arm's from the very first step: at iter 20000
# a 160k run's LR is still ~8.9e-5 while the 20k arm had already annealed to ~0.
# It answers "what does this decoder reach when given the budget", and its val
# curve must not be read as the 20k arm's curve continued.  `resume60k` is the
# opposite: it keeps the 20k arm's trajectory and re-anchors the LR so that step
# 20000+n matches a 60k-from-scratch curve pointwise (see below).  Do not mix them.
#
# The `long` arm's work dir is derived from the budget, e.g. H2L_MAX_ITERS=160000
# -> runs/mask2former_uav/h2_swin_l_mask2former_160k_seed3407_fp32.
#
# Why `resume60k` is not a plain `--resume`
# -----------------------------------------
# The 20k checkpoint stores the PolyLR state (MMEngine saves param schedulers
# even with `save_optimizer=False`).  That state pins `end=20000`, and
# `_ParamScheduler.step` only updates the LR while `_global_step < end`, so a
# naive resume would never touch the LR again -- 40k iterations at a frozen
# value.  `prepare_resume_checkpoint.py` therefore rewrites the checkpoint with
# the scheduler state dropped, and the LR comes from `--base-lr/--scheduler-end`
# (a re-anchored PolyLR that reproduces a from-scratch 60k curve exactly).
#
# Two deterministic gates guard the 40k-iteration run: the *dumped, resolved*
# config must contain the four overrides, and the run must report
# `resumed epoch: 0, iter: 20000`.  Either failure kills the process group.
set -euo pipefail
mode="${1:-full}"
case "$mode" in
    smoke|full|long|resume60k) ;;
    *) echo "usage: $0 [smoke|full|long|resume60k]"; exit 64 ;;
esac
project_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$project_root"
export PYTHONPATH="$project_root/../mmsegmentation:$project_root${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

python_bin="$project_root/.venv-mask2former/bin/python"
backbone="$project_root/models/swin_l/swin_large_patch4_window12_384_22k_20220412-6580f57d.pth"
val_interval="${H2_VAL_INTERVAL:-1000}"
max_iters="${H2_MAX_ITERS:-20000}"
extra_cfg=()
if [[ "${H2_WITH_CP:-0}" == "1" ]]; then
    extra_cfg=(--extra-cfg-option model.backbone.with_cp=True)
    echo "note: Swin gradient checkpointing enabled via H2_WITH_CP=1"
fi

# ---- preflight: CUDA must be real, not just a device node ----
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
"$python_bin" -c 'import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)' \
    || { echo "ABORT: torch.cuda.is_available() is False; restart the instance in the AutoDL console"; exit 2; }
[[ -f "$backbone" ]] || { echo "ABORT: missing Swin-L backbone $backbone"; exit 2; }
sha256sum "$backbone"
"$python_bin" experiment/mask2former_uav/run.py check \
    --data-root "$project_root/dataset/low_altitude_2026" \
    --split-dir "$project_root/runs/splits"

run_train() {  # $1 = work dir, remaining args = run.py train flags
    local work_dir="$1"; shift
    mkdir -p "$work_dir"
    date -Is
    sha256sum experiment/mask2former_uav/run.py \
              experiment/mask2former_uav/mask2former_swin_l_512.py | tee "$work_dir/code_hashes.txt"
    "$python_bin" -u experiment/mask2former_uav/run.py train \
        --model mask2former_swin_l \
        --mmseg-root "$project_root/../mmsegmentation" \
        --data-root "$project_root/dataset/low_altitude_2026" \
        --split-dir "$project_root/runs/splits" \
        --work-dir "$work_dir" \
        --backbone-checkpoint "$backbone" \
        --batch-size 2 --accumulative-counts 1 --num-workers 4 --seed 3407 \
        ${extra_cfg[@]+"${extra_cfg[@]}"} "$@"
}

smoke_dir="$project_root/runs/mask2former_uav/h2_smoke_seed3407"

if [[ "$mode" == "smoke" ]]; then
    rm -rf "$smoke_dir"
    run_train "$smoke_dir" --max-iters 20 --val-interval 20 --max-keep-ckpts 1 --no-save-optimizer
    echo "SMOKE OK"
    exit 0
fi

if [[ "$mode" == "resume60k" ]]; then
    # ------------------------------------------------------------------
    # 1. locate and rewrite the source checkpoint (idempotent)
    # ------------------------------------------------------------------
    total_iters="${H2B_TOTAL_ITERS:-60000}"
    resumed_iter="${H2B_RESUMED_ITER:-20000}"
    src_dir="$project_root/runs/mask2former_uav/h2_swin_l_mask2former_20k_seed3407_fp32"
    src_ckpt="${H2B_SRC_CKPT:-}"
    if [[ -z "$src_ckpt" ]]; then
        for candidate in "$src_dir/iter_${resumed_iter}.pth" \
                         "$src_dir/best_mIoU_iter_${resumed_iter}.pth"; do
            [[ -f "$candidate" ]] && { src_ckpt="$candidate"; break; }
        done
    fi
    # Fail closed: without a source checkpoint there is nothing to resume, and
    # MMSeg's `--resume` with an empty work dir silently starts from scratch.
    [[ -n "$src_ckpt" && -f "$src_ckpt" ]] || {
        echo "ABORT: no checkpoint at iter $resumed_iter under $src_dir"
        echo "       (looked for iter_${resumed_iter}.pth and best_mIoU_iter_${resumed_iter}.pth)"
        exit 2
    }

    work_dir="$project_root/runs/mask2former_uav/h2_swin_l_mask2former_20k_to_60k_seed3407_fp32"
    prep_dir="$project_root/runs/mask2former_uav/_resume_prep"
    stripped="$prep_dir/h2_iter${resumed_iter}_nosched.pth"
    plan="$prep_dir/h2_to_60k_plan.json"

    "$python_bin" experiment/mask2former_uav/prepare_resume_checkpoint.py --self-test
    if [[ ! -f "$stripped" || ! -f "$plan" ]]; then
        mkdir -p "$prep_dir"
        "$python_bin" experiment/mask2former_uav/prepare_resume_checkpoint.py \
            --source "$src_ckpt" --output "$stripped" --plan "$plan" \
            --total-iters "$total_iters" --expect-iter "$resumed_iter"
    else
        echo "reusing prepared checkpoint $stripped"
    fi
    read -r base_lr sched_end < <("$python_bin" - "$plan" <<'PY'
import json, sys
plan = json.load(open(sys.argv[1]))
print(repr(plan['base_lr']), plan['scheduler_end'])
PY
)
    echo "resume plan: source=$src_ckpt base_lr=$base_lr scheduler_end=$sched_end total_iters=$total_iters"

    mkdir -p "$work_dir"
    exec 9>"$work_dir/train.lock"
    if command -v flock >/dev/null 2>&1; then
        flock -n 9 || { echo 'H2b is already running'; exit 1; }
    else
        echo "WARNING: flock unavailable; duplicate-start guard disabled"
    fi
    if [[ -f "$work_dir/last_checkpoint" || -f "$work_dir/exit_code" ]]; then
        echo 'Work directory already contains a run; inspect it before restarting.'
        exit 1
    fi
    trap 'result=$?; printf "%s\n" "$result" > "$work_dir/exit_code"' EXIT

    # Checkpoints with optimizer state are ~2.6 GB instead of ~0.9 GB.  Keeping
    # the optimizer is what makes *this* run resumable mid-way (its own `end`
    # matches the config, so the scheduler state is not stale), so only fall
    # back when the system disk is genuinely tight.
    save_opt_flag=(--save-optimizer)
    avail_kb=$(df -Pk "$work_dir" | awk 'NR==2 {print $4}')
    if (( avail_kb < 12 * 1024 * 1024 )); then
        echo "WARNING: only $((avail_kb / 1024 / 1024)) GiB free on this filesystem;"
        echo "         falling back to --no-save-optimizer (checkpoints ~0.9 GB)"
        save_opt_flag=(--no-save-optimizer)
    else
        echo "disk headroom: $((avail_kb / 1024 / 1024)) GiB free; keeping optimizer state"
    fi

    # ------------------------------------------------------------------
    # 2. start the continuation detached, then verify it before trusting it
    # ------------------------------------------------------------------
    echo "=== starting 20k->60k: max_iters=$total_iters val_interval=$val_interval ==="
    setsid "$python_bin" -u experiment/mask2former_uav/run.py train \
        --model mask2former_swin_l \
        --mmseg-root "$project_root/../mmsegmentation" \
        --data-root "$project_root/dataset/low_altitude_2026" \
        --split-dir "$project_root/runs/splits" \
        --work-dir "$work_dir" \
        --backbone-checkpoint "$backbone" \
        --batch-size 2 --accumulative-counts 1 --num-workers 4 --seed 3407 \
        --max-iters "$total_iters" --val-interval "$val_interval" \
        --scheduler-end "$sched_end" --base-lr "$base_lr" \
        --resume-from "$stripped" \
        --max-keep-ckpts 2 "${save_opt_flag[@]}" \
        ${extra_cfg[@]+"${extra_cfg[@]}"} \
        > "$work_dir/train.log" 2>&1 &
    train_pid=$!

    abort_run() {
        echo "ABORT: $1"
        printf '4\n' > "$work_dir/exit_code"
        # `setsid` normally execs in place, so $train_pid is the new process-group
        # leader and `kill -- -$train_pid` takes the whole tree (run.py plus the
        # tools/train.py it spawned).  If that assumption does not hold, a silent
        # failure would leave a wrongly-configured run alive, so cover both.
        pkill -TERM -P "$train_pid" 2>/dev/null || true
        kill -TERM "$train_pid" 2>/dev/null || true
        pgid=$(ps -o pgid= -p "$train_pid" 2>/dev/null | tr -d ' ')
        if [[ -n "$pgid" && "$pgid" == "$train_pid" ]]; then
            kill -TERM -- "-$pgid" 2>/dev/null || true
        else
            echo "note: $train_pid is not its own process-group leader (pgid=${pgid:-gone});" \
                 "killed it and its children directly"
        fi
        sleep 5
        pkill -KILL -P "$train_pid" 2>/dev/null || true
        kill -KILL "$train_pid" 2>/dev/null || true
        if kill -0 "$train_pid" 2>/dev/null; then
            echo "WARNING: $train_pid survived SIGKILL; check for a leftover tools/train.py"
        fi
        exit 4
    }

    # gate 1: the *resolved* config must carry the four overrides
    dumped="$work_dir/mask2former_swin_l_512.py"
    for _ in $(seq 1 40); do [[ -f "$dumped" ]] && break; sleep 3; done
    [[ -f "$dumped" ]] || abort_run "the resolved config was never dumped to $dumped"
    for want in "max_iters=$total_iters" "lr=$base_lr" "end=$sched_end"; do
        grep -q -- "$want" "$dumped" \
            || abort_run "'$want' is missing from the resolved config $dumped"
    done
    echo "gate 1 OK: resolved config carries max_iters=$total_iters lr=$base_lr end=$sched_end"

    # gate 2: the iteration counter must really be restored from the checkpoint
    resumed_marker="resumed epoch:.*iter: $resumed_iter"
    found=0
    for _ in $(seq 1 60); do
        if grep -qE "$resumed_marker" "$work_dir/train.log"; then found=1; break; fi
        if ! kill -0 "$train_pid" 2>/dev/null; then break; fi
        sleep 5
    done
    if [[ "$found" != "1" ]]; then
        tail -n 20 "$work_dir/train.log" || true
        abort_run "the run never reported '$resumed_marker'; it is not the continuation we intended"
    fi
    grep -m1 -E "$resumed_marker" "$work_dir/train.log"
    echo "gate 2 OK: iteration counter resumed at $resumed_iter"

    # gate 3 (advisory): the first logged LR should be the anchored value
    for _ in $(seq 1 60); do
        scalars=$(ls -t "$work_dir"/*/vis_data/scalars.json 2>/dev/null | head -n1)
        if [[ -n "$scalars" ]] && grep -q '"lr"' "$scalars"; then
            last_lr_line=$(grep '"lr"' "$scalars" | tail -n1)
            "$python_bin" - "$last_lr_line" "$base_lr" <<'PY' || true
import json, sys
expected = float(sys.argv[2])
value = json.loads(sys.argv[1])['lr']
values = [float(v) for v in (value if isinstance(value, list) else [value])]
observed = max(values)
rel = abs(observed - expected) / expected
kind = 'OK' if rel < 0.02 else 'WARNING: does not match the anchor'
print(f'gate 3 {kind}: observed lr={observed:.6e} over {len(values)} group(s), '
      f'anchor={expected:.6e}, relative difference {rel:.2%}')
PY
            break
        fi
        kill -0 "$train_pid" 2>/dev/null || break
        sleep 5
    done

    wait "$train_pid" || true
    date -Is
    echo "H2b DONE"
    exit 0
fi

if [[ "$mode" == "long" ]]; then
    max_iters="${H2L_MAX_ITERS:-160000}"
    val_interval="${H2L_VAL_INTERVAL:-2000}"
    label="$((max_iters / 1000))k"
    work_dir="$project_root/runs/mask2former_uav/h2_swin_l_mask2former_${label}_seed3407_fp32"
else
    work_dir="$project_root/runs/mask2former_uav/h2_swin_l_mask2former_20k_seed3407_fp32"
    label="20k"
fi
mkdir -p "$work_dir"
exec 9>"$work_dir/train.lock"
if command -v flock >/dev/null 2>&1; then
    flock -n 9 || { echo "H2 ($label) is already running"; exit 1; }
else
    echo "WARNING: flock unavailable; duplicate-start guard disabled"
fi

# The long arm is a ~15 h run and GPUs do vanish mid-run on this platform, so keep
# the Adam state and let `H2L_RESUME=1` pick the run back up when it does.  Its own
# scheduler state carries `end=160000`, which matches the config, so a plain
# `--resume` is safe here -- unlike the `resume60k` case, where the stale `end` of
# a finished 20k run would silently freeze the LR.
# Checkpoints are ~2.6 GB with optimizer state and ~0.88 GB without, and `best`
# does not consume a `max_keep_ckpts` slot.
resume_flag=()
if [[ "$mode" == "long" && "${H2L_RESUME:-0}" == "1" ]]; then
    [[ -f "$work_dir/last_checkpoint" ]] || {
        echo "ABORT: H2L_RESUME=1 but $work_dir has no last_checkpoint to resume from"
        exit 2
    }
    resume_flag=(--resume)
    echo "resuming $label from $(cat "$work_dir/last_checkpoint")"
elif [[ -f "$work_dir/last_checkpoint" ]]; then
    echo 'Work directory already contains a run; inspect it before restarting.'
    exit 1
fi

avail_kb=$(df -Pk "$work_dir" | awk 'NR==2 {print $4}')
if (( avail_kb >= 12 * 1024 * 1024 )); then
    save_opt_flag=(--save-optimizer)
    max_keep=1
else
    echo "WARNING: only $((avail_kb / 1024 / 1024)) GiB free on this filesystem;"
    echo "         falling back to --no-save-optimizer (checkpoints ~0.88 GB)"
    save_opt_flag=(--no-save-optimizer)
    max_keep=3
fi
trap 'result=$?; printf "%s\n" "$result" > "$work_dir/exit_code"' EXIT
echo "checkpoint policy: ${save_opt_flag[0]} max_keep_ckpts=$max_keep, $((avail_kb / 1024 / 1024)) GiB free"

# ---- gate 1: smoke test. Cheap insurance against burning hours on a broken run. ----
if [[ "$mode" != "long" || "${H2L_RESUME:-0}" != "1" ]]; then
    rm -rf "$smoke_dir"
    run_train "$smoke_dir" --max-iters 20 --val-interval 20 --max-keep-ckpts 1 --no-save-optimizer \
        || { echo "ABORT: smoke test failed; not starting the $label run"; exit 3; }
    echo "smoke validation metrics:"
    cat "$smoke_dir/val_metrics.json" 2>/dev/null || echo "(no val_metrics.json written)"
    rm -rf "$smoke_dir"
fi

# ---- gate 2: the real run ----
echo "=== starting $label: val_interval=$val_interval max_iters=$max_iters ==="
run_train "$work_dir" --max-iters "$max_iters" --val-interval "$val_interval" \
    --max-keep-ckpts "$max_keep" "${save_opt_flag[@]}" \
    ${resume_flag[@]+"${resume_flag[@]}"}
date -Is
echo "H2 ($label) DONE"
