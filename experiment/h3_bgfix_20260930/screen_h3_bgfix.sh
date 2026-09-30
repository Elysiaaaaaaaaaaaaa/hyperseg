#!/usr/bin/env bash
# Sequential screening driver for the H3 background-collapse fix.
#
#   bash experiment/h3_bgfix_20260930/screen_h3_bgfix.sh 20000 1000
#   bash experiment/h3_bgfix_20260930/screen_h3_bgfix.sh 20000 1000 baseline loss
#
# $1 = per-arm iteration budget, $2 = validation interval, remaining args = the arms to run
# (default: all four).  Each arm is started through launch_h3_bgfix.sh, which already runs its
# own smoke test and refuses to clobber a finished run, and this driver waits for that arm's
# exit_code to appear before starting the next one so the GPU is never oversubscribed.
#
# Screening runs use --drop-optimizer-state: Swin-L spends 1.47 GiB on AdamW state against
# 0.74 GiB of weights, and four arms keeping both would not fit the 11 GiB free on /.
#
# The driver itself is detached, so a dropped SSH connection cannot interrupt the sequence:
#   nohup bash experiment/h3_bgfix_20260930/screen_h3_bgfix.sh 20000 1000 > screen.log 2>&1 &
set -uo pipefail

budget="${1:-20000}"
val_interval="${2:-1000}"
if [[ $# -ge 2 ]]; then shift 2; elif [[ $# -ge 1 ]]; then shift 1; fi
arms=("$@")
if [[ ${#arms[@]} -eq 0 ]]; then arms=(baseline loss aug full); fi

project_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$project_root"

suffix=$(( budget / 1000 ))
driver_log="$project_root/runs/mask2former_uav/h3_bgfix_screen.log"

log() { printf '%s  %s\n' "$(date -Is)" "$*" | tee -a "$driver_log"; }

log "screening start: budget=$budget val_interval=$val_interval arms=${arms[*]}"

for arm in "${arms[@]}"; do
    tag="${arm}_${suffix}k"
    work_dir="$project_root/runs/mask2former_uav/h3_bgfix_${tag}_seed3407"

    if [[ -f "$work_dir/exit_code" ]] && [[ "$(head -n 1 "$work_dir/exit_code")" == "0" ]]; then
        log "SKIP $arm: $work_dir already finished successfully"
        continue
    fi
    if [[ -f "$work_dir/config.json" ]]; then
        log "ABORT before $arm: $work_dir holds an unfinished run; inspect it by hand"
        exit 1
    fi

    log "START $arm  ->  $work_dir"
    if ! bash experiment/h3_bgfix_20260930/launch_h3_bgfix.sh \
            "$arm" "$tag" --max-iters "$budget" --val-interval "$val_interval" \
            --drop-optimizer-state >> "$driver_log" 2>&1; then
        log "ABORT: launcher refused to start $arm"
        exit 1
    fi

    # Wait for the detached trainer.  launch_h3_bgfix.sh writes exit_code only when the
    # wrapper finishes, so its absence means the arm is still running.
    while [[ ! -f "$work_dir/exit_code" ]]; do
        sleep 60
        if [[ -f "$work_dir/train.pid" ]] && ! kill -0 "$(cat "$work_dir/train.pid")" 2>/dev/null \
                && [[ ! -f "$work_dir/exit_code" ]]; then
            sleep 30
            [[ -f "$work_dir/exit_code" ]] || log "WARNING: $arm pid is gone but no exit_code yet"
        fi
    done

    status=$(head -n 1 "$work_dir/exit_code")
    if [[ "$status" != "0" ]]; then
        log "ABORT: $arm exited $status; see $work_dir/train.log"
        exit 1
    fi
    log "DONE $arm  ($(grep -c '"iter"' "$work_dir/val_history.jsonl" 2>/dev/null || echo 0) validations)"
done

log "screening complete"
