#!/usr/bin/env bash
# Server-side progress logger for the H2 Swin-L + Mask2Former runs.
#
#   cd /root/hyperseg
#   setsid nohup bash experiment/mask2former_uav/watch_h2_remote.sh 300 \
#       >/dev/null 2>&1 < /dev/null &
#
#   # a specific arm:
#   setsid nohup bash experiment/mask2former_uav/watch_h2_remote.sh 300 \
#       runs/mask2former_uav/h2_swin_l_mask2former_20k_to_60k_seed3407_fp32 h2b \
#       runs/mask2former_uav/h2b_watch.log >/dev/null 2>&1 < /dev/null &
#
# Same rationale and same line shape as the bgfix watcher (h3_bgfix_20260930/watch_remote.sh):
# training is detached, but a watcher on the laptop dies with the laptop, so the history lives
# on the server and is downloaded afterwards.  Lines read
#
#   2026-09-30 21:20:00  h2=running[iter 1234|v1|lr 9.90e-05]
#
# `iter` and `v` keep their positions/order for anything that parses the older lines; `lr` is
# appended because a *resumed* run's LR is the first thing that can be silently wrong.
#
# Three MMSeg-specific differences from the hyperSeg watcher:
#   * iteration comes from the ``[cur/max]`` marker in the run's own text log;
#   * the number of completed validations comes from ``vis_data/scalars.json`` (MMEngine writes
#     one validation record per validation), because MMSeg keeps no val_history.jsonl.
#     The key is the **bare** ``mIoU`` -- MMSeg 1.2.2 does not write ``val/mIoU``; only the
#     validation rows carry it, so counting occurrences of ``"mIoU"`` is exact.
#   * LR comes from the same scalars file (the max over the param groups, since the backbone
#     group runs at 0.1x).
#
# MMSeg writes that text log to ``<work_dir>/<timestamp>/<timestamp>.log`` -- there is no
# ``<work_dir>/train.log``, so the newest file in the timestamp directory is used.
#
# It exits by itself once the run writes its exit_code, so it does not linger forever.
set -uo pipefail

interval="${1:-300}"
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
runs="$root/runs/mask2former_uav"
work_dir="${2:-$runs/h2_swin_l_mask2former_20k_seed3407_fp32}"
label="${3:-h2}"
log="${4:-${H2_WATCH_LOG:-$runs/h2_watch.log}}"
case "$work_dir" in /*) ;; *) work_dir="$root/$work_dir" ;; esac
lock="$log.lock"
mkdir -p "$(dirname "$log")"

# flock is not guaranteed present, and an unconditional `flock -n` on a box without it makes the
# guard itself fail closed -- the script would exit having written nothing, which looks exactly
# like "the run ended".  Degrade loudly instead of silently.
if command -v flock >/dev/null 2>&1; then
    exec 9>"$lock"
    if ! flock -n 9; then
        echo "another watcher already holds $lock; exiting"
        exit 0
    fi
else
    echo "warning: flock not found; duplicate watchers will both append to $log" >&2
fi

printf '%s  watcher started (interval %ss, %s, dir %s)\n' \
    "$(date '+%Y-%m-%d %H:%M:%S')" "$interval" "$label" "$work_dir" >> "$log"

while true; do
    if [ ! -d "$work_dir" ]; then
        printf '%s  %s=not started[iter -|v0|lr -]\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$label" >> "$log"
        sleep "$interval"
        continue
    fi

    iter="-"
    text_log=$(ls -t "$work_dir"/*/*.log 2>/dev/null | head -n1)
    if [ -n "$text_log" ]; then
        # MMSeg right-aligns the counter to the width of the budget, so a 160k run logs
        # `[    200/160000]`, not `[200/160000]` -- the spaces are part of the field.
        found=$(grep -o '\[ *[0-9]\+/ *[0-9]\+ *\]' "$text_log" 2>/dev/null \
                | tail -n1 | sed -e 's/.*\[ *//' -e 's/ *\/.*//')
        [ -n "$found" ] && iter="$found"
    fi
    scalars=$(ls -t "$work_dir"/*/vis_data/scalars.json 2>/dev/null | head -n1)
    vals=0
    lr="-"
    if [ -n "$scalars" ]; then
        vals=$(grep -c '"mIoU"' "$scalars" 2>/dev/null | tr -d ' ')
        # `lr` is a list with one entry per param group (the backbone runs at 0.1x), so report
        # the largest rather than the first one.  Extract the array itself first: the same JSON
        # line also carries `memory`, and a plain "everything after `\"lr\":`" split would pick
        # that number up as the largest value.
        lr=$(grep -o '"lr": *\[[^]]*\]' "$scalars" 2>/dev/null | tail -n1 \
             | grep -oE '[0-9]+\.?[0-9]*([eE][-+]?[0-9]+)?' \
             | sort -g | tail -n1)
        [ -n "$lr" ] || lr="-"
    fi

    settled=0
    if [ -f "$work_dir/exit_code" ]; then
        code=$(head -n1 "$work_dir/exit_code")
        if [ "$code" = "0" ]; then
            status="done"
        else
            status="failed($code)"
        fi
        settled=1
    elif [ "$iter" != "-" ]; then
        status="running"
    else
        status="starting"
    fi

    printf '%s  %s=%s[iter %s|v%s|lr %s]\n' "$(date '+%Y-%m-%d %H:%M:%S')" \
        "$label" "$status" "$iter" "$vals" "$lr" >> "$log"

    # An exit code is the only completion signal; a pid check would lie after a reboot.
    if [ "$settled" -eq 1 ]; then
        printf '%s  %s settled\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$label" >> "$log"
        break
    fi
    sleep "$interval"
done
