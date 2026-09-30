#!/usr/bin/env bash
# Concurrent screening driver for the H3 background-collapse fix.
#
#   nohup bash experiment/h3_bgfix_20260930/screen_parallel_h3_bgfix.sh \
#       20000 1000 2 loss aug full > runs/mask2former_uav/h3_bgfix_screen.log 2>&1 &
#
# $1 = per-arm iteration budget, $2 = validation interval, $3 = how many arms may run at once,
# remaining args = the arms to run.
#
# Why a slot limit instead of "just run them all": Swin-L at 512/batch-2 with gradient
# checkpointing holds about 5.2 GiB of the 24 GiB card, so four concurrent arms leave under
# 4 GiB of headroom and one activation spike takes down every arm at once.
#
# Measured on a 4090: one process runs at 0.194 s/update and already holds 96% GPU
# utilisation; three concurrent processes run at 0.623 s/update each, an aggregate of
# 4.82 updates/s against 5.15 solo.  So the slot count is about staying inside memory, NOT
# about going faster -- this model saturates the card on its own and concurrency trades
# roughly even.  Keep the limit at 2-3 for safety and expect the schedule to be set by the
# total update count divided by the solo rate.
#
# The arms train on identical batches in identical order -- shuffling and augmentation are
# seeded, and no scheduling decision depends on wall time -- so running them concurrently
# does not disturb the comparison.
#
# Screening runs use --drop-optimizer-state: Swin-L keeps 1.47 GiB of AdamW state against
# 0.74 GiB of weights, and four arms holding both would not fit the 11 GiB free on /.
set -uo pipefail

budget="${1:-20000}"
val_interval="${2:-1000}"
slots="${3:-2}"
if [[ $# -ge 3 ]]; then shift 3; elif [[ $# -ge 2 ]]; then shift 2; elif [[ $# -ge 1 ]]; then shift 1; fi
arms=("$@")
if [[ ${#arms[@]} -eq 0 ]]; then arms=(baseline loss aug full); fi

project_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$project_root"

suffix=$(( budget / 1000 ))
driver_log="$project_root/runs/mask2former_uav/h3_bgfix_screen.log"
mkdir -p "$(dirname "$driver_log")"
# Overridable so the supervision loop can be exercised against stub launchers.
poll_seconds="${H3BG_POLL_SECONDS:-45}"

log() { printf '%s  %s\n' "$(date -Is)" "$*" | tee -a "$driver_log"; }
# The launcher names its work directory from the tag, so the tag is built in exactly one
# place here and reused for both the launch call and the completion check.
tag_for() { printf '%s_%dk' "$1" "$suffix"; }
work_dir_for() { printf '%s/runs/mask2former_uav/h3_bgfix_%s_seed3407' "$project_root" "$(tag_for "$1")"; }

log "parallel screening start: budget=$budget val_interval=$val_interval slots=$slots arms=${arms[*]}"

queue=("${arms[@]}")
active=()
failed=0

while [[ ${#queue[@]} -gt 0 || ${#active[@]} -gt 0 ]]; do
    while [[ ${#active[@]} -lt $slots && ${#queue[@]} -gt 0 ]]; do
        arm="${queue[0]}"
        queue=("${queue[@]:1}")
        work_dir=$(work_dir_for "$arm")

        if [[ -f "$work_dir/exit_code" ]] && [[ "$(head -n 1 "$work_dir/exit_code")" == "0" ]]; then
            log "SKIP $arm: already finished successfully"
            continue
        fi
        if [[ -f "$work_dir/config.json" && ! -f "$work_dir/exit_code" ]]; then
            log "NOTE $arm: an unfinished run already owns $work_dir; adopting it rather than"
            log "          starting a second copy.  If that run is actually dead (the box was"
            log "          rebooted, so no exit_code was ever written), move the directory aside."
            active+=("$arm")
            continue
        fi
        if [[ -f "$work_dir/config.json" ]]; then
            log "SKIP $arm: $work_dir holds a failed run; inspect it before re-running"
            failed=1
            continue
        fi

        log "START $arm -> $work_dir"
        if ! bash experiment/h3_bgfix_20260930/launch_h3_bgfix.sh \
                "$arm" "$(tag_for "$arm")" --max-iters "$budget" --val-interval "$val_interval" \
                --drop-optimizer-state >> "$driver_log" 2>&1; then
            log "FAILED to launch $arm (see $driver_log)"
            failed=1
            continue
        fi
        active+=("$arm")
    done

    # A slot frees up when its arm writes exit_code, which the detached wrapper emits on exit.
    still_active=()
    for arm in "${active[@]}"; do
        work_dir=$(work_dir_for "$arm")
        if [[ -f "$work_dir/exit_code" ]]; then
            status=$(head -n 1 "$work_dir/exit_code")
            validations=$(wc -l < "$work_dir/val_history.jsonl" 2>/dev/null || echo 0)
            if [[ "$status" == "0" ]]; then
                log "DONE $arm after $validations validations"
            else
                log "FAILED $arm with exit status $status; see $work_dir/train.log"
                failed=1
            fi
        else
            still_active+=("$arm")
        fi
    done
    active=(${still_active[@]+"${still_active[@]}"})

    [[ ${#active[@]} -gt 0 || ${#queue[@]} -gt 0 ]] && sleep "$poll_seconds"
done

if [[ $failed -ne 0 ]]; then
    log "screening finished WITH FAILURES"
    exit 1
fi
log "screening complete: all arms finished"
