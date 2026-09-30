#!/usr/bin/env bash
# Server-side progress logger for the H3 bgfix arms.
#
#   cd /root/hyperseg
#   nohup bash experiment/h3_bgfix_20260930/watch_remote.sh 20k 300 baseline loss aug full \
#       >/dev/null 2>&1 &
#
# Why this exists: the training is detached on the server, but a watcher running on the laptop
# dies with the laptop.  Keeping the log on the server means the whole history can simply be
# downloaded afterwards, and nothing is lost if the local machine is closed overnight.
#
# The line format is deliberately identical to what ``watch_screen.py`` writes, so a downloaded
# copy can be fed straight to ``eta_screen.py --log <file>`` and to the same parsers.
#
#   usage: watch_remote.sh [suffix] [interval_seconds] [arms...]
#
# It exits by itself once every arm has an exit code, so it does not linger forever.
set -uo pipefail

suffix="${1:-20k}"
interval="${2:-300}"
shift 2 2>/dev/null || true
arms=("$@")
if [ "${#arms[@]}" -eq 0 ]; then
    arms=(baseline loss aug full)
fi

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
runs="$root/runs/mask2former_uav"
log="${H3BG_WATCH_LOG:-$runs/h3_bgfix_watch.log}"
lock="$log.lock"

# One writer per log, so a restart cannot produce interleaved or duplicated lines.
# flock is not guaranteed present, and an unconditional `flock -n` on a box without it makes the
# guard itself fail closed -- the script would exit having written nothing, which looks exactly
# like "the run ended".  So degrade loudly instead of silently.
if command -v flock >/dev/null 2>&1; then
    exec 9>"$lock"
    if ! flock -n 9; then
        echo "another watcher already holds $lock; exiting"
        exit 0
    fi
else
    echo "warning: flock not found; duplicate watchers will both append to $log" >&2
fi

printf '%s  watcher started (interval %ss, suffix %s, arms: %s)\n' \
    "$(date '+%Y-%m-%d %H:%M:%S')" "$interval" "$suffix" "${arms[*]}" >> "$log"

while true; do
    line="$(date '+%Y-%m-%d %H:%M:%S') "
    settled=0
    for arm in "${arms[@]}"; do
        work_dir="$runs/h3_bgfix_${arm}_${suffix}_seed3407"
        iter="-"
        vals=""
        if [ ! -d "$work_dir" ]; then
            status="not started"
        else
            found=$(grep -o '^iter [0-9]*' "$work_dir/train.log" 2>/dev/null | tail -n1 | awk '{print $2}')
            [ -n "$found" ] && iter="$found"
            # Braces + redirect: an unreadable path makes the *shell* complain, and a plain
            # `2>/dev/null` on the command does not silence the shell's own message.
            vals=$( { wc -l < "$work_dir/val_history.jsonl"; } 2>/dev/null | tr -d ' ')
            if [ -f "$work_dir/exit_code" ]; then
                code=$(head -n1 "$work_dir/exit_code")
                if [ "$code" = "0" ]; then
                    status="done"
                else
                    status="failed($code)"
                fi
                settled=$((settled + 1))
            elif [ "$iter" != "-" ] || [ -n "$vals" ]; then
                status="running"
            else
                status="not started"
            fi
        fi
        line="${line} ${arm}=${status}[iter ${iter}|v${vals}]"
    done
    printf '%s\n' "$line" >> "$log"

    # Every arm has reported an exit code: nothing left to observe, so stop rather than linger.
    # A failed arm also ends the wait -- its log is the evidence, and the driver has already
    # aborted by then.
    if [ "$settled" -ge "${#arms[@]}" ]; then
        printf '%s  all arms settled\n' "$(date '+%Y-%m-%d %H:%M:%S')" >> "$log"
        break
    fi
    sleep "$interval"
done
