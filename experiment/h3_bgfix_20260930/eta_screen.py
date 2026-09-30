"""Estimate when each arm of the H3 bgfix screening run will finish.

    python experiment/h3_bgfix_20260930/eta_screen.py

Two things have to be measured rather than assumed, because both surprised us once already:

* the effective seconds-per-update, which includes the interleaved validations and therefore
  creeps away from the ``sec_per_update`` printed in the training log as validation time is
  amortised (0.194 s/update measured solo, 0.79 s/update effective with three arms running);
* which arms are actually running, since a queued arm's start time depends on the first slot
  freeing up, not on the wall clock.

The rate comes from the timestamped snapshots that ``watch_screen.py`` appends to
``logs/watch_screen.log``: differencing two samples cancels any constant clock offset between
this machine and the server, so the rate is trustworthy even if the clocks disagree.  The
anchor -- current iteration and "now" -- is read from the server itself.

A queued arm cannot be measured, so its duration is modelled from ``--solo-rate`` (the effective
rate when the GPU is not shared).  That is an assumption and is labelled as such in the output.
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import remote  # noqa: E402

LINE = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\s+(.*)$")
ARM = re.compile(r"(\w+)=([^\[]*)\[([^|\]]*)\|v(\d*)\]")


def read_history(path: Path) -> dict[str, list[tuple[float, int]]]:
    """Return ``{arm: [(epoch, iter), ...]}`` from the watcher's log, oldest first."""
    history: dict[str, list[tuple[float, int]]] = {}
    if not path.exists():
        return history
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        matched = LINE.match(line.strip())
        if not matched:
            continue
        stamp = time.mktime(time.strptime(matched.group(1), "%Y-%m-%d %H:%M:%S"))
        for arm, _status, iteration, _count in ARM.findall(matched.group(2)):
            # The iteration field is the literal text "iter 5200", and a queued arm carries "-",
            # so the number has to be extracted rather than tested for directly.
            digits = re.sub(r"\D", "", iteration)
            if digits:
                history.setdefault(arm, []).append((stamp, int(digits)))
    return history


def rate_for(samples: list[tuple[float, int]], window: int) -> tuple[float, float] | None:
    """Seconds per update over the most recent ``window`` samples, plus the span in seconds."""
    tail = samples[-window:]
    if len(tail) < 2:
        return None
    (t0, i0), (t1, i1) = tail[0], tail[-1]
    if i1 <= i0 or t1 <= t0:
        return None
    return (t1 - t0) / (i1 - i0), t1 - t0


def remote_state(client, arms, suffix, budget):
    fields = {}
    for arm in arms:
        work_dir = f"/root/hyperseg/runs/mask2former_uav/h3_bgfix_{arm}_{suffix}_seed3407"
        _, text = remote.run(
            client,
            f'echo "DATE=$(date +%s)"; '
            f'echo "EXIT_{arm}=$(cat {work_dir}/exit_code 2>/dev/null | head -n1)"; '
            f'echo "ITER_{arm}=$(grep -o \'^iter [0-9]*\' {work_dir}/train.log 2>/dev/null '
            f'| tail -n1 | awk \'{{print $2}}\')"; '
            f'echo "VALS_{arm}=$(wc -l < {work_dir}/val_history.jsonl 2>/dev/null | tr -d " ")"',
            timeout=120,
        )
        for line in text.splitlines():
            key, separator, value = line.partition("=")
            if separator:
                fields[key.strip()] = value.strip()
    now = int(fields.get("DATE") or time.time())
    state = {}
    for arm in arms:
        exit_code = fields.get(f"EXIT_{arm}", "")
        iteration = fields.get(f"ITER_{arm}", "")
        state[arm] = {
            "exit": exit_code,
            "iter": int(iteration) if iteration.isdigit() else None,
            "vals": fields.get(f"VALS_{arm}", ""),
            "done": exit_code == "0",
            "failed": bool(exit_code) and exit_code != "0",
        }
    return now, state


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--server", default="server2", choices=("server1", "server2", "server3"))
    parser.add_argument("--suffix", default="20k")
    parser.add_argument("--arms", nargs="+", default=["baseline", "loss", "aug", "full"])
    parser.add_argument("--budget", type=int, default=20000, help="target iterations per arm")
    parser.add_argument("--log", type=Path, default=HERE / "logs/watch_screen.log")
    parser.add_argument("--window", type=int, default=12, help="snapshots used for the rate")
    parser.add_argument("--solo-rate", type=float, default=0.25,
                        help="assumed effective seconds/update for a queued arm running alone")
    parser.add_argument("--slots", type=int, default=2, help="concurrent arms the driver allows")
    parser.add_argument("--driver-arms", nargs="+", default=["loss", "aug", "full"],
                        help="arms the slot driver manages; an arm started outside it can finish "
                             "without releasing a slot")
    args = parser.parse_args()

    history = read_history(args.log)
    if not history:
        print(f"no usable snapshots in {args.log}; start watch_screen.py first", file=sys.stderr)
        return 2

    client = remote.connect(args.server)
    try:
        now, state = remote_state(client, args.arms, args.suffix, args.budget)
    finally:
        client.close()

    rates: dict[str, float] = {}
    print("arm        iter/budget    rate s/upd   source            eta")
    print("-" * 74)
    for arm in args.arms:
        info = state[arm]
        if info["done"]:
            print(f"{arm:<10} {args.budget:>6}/{args.budget}   -            finished          done")
            continue
        if info["failed"]:
            print(f"{arm:<10} {'':>6}        -            failed            "
                  f"exit={info['exit']}")
            continue
        measured = rate_for(history.get(arm, []), args.window)
        if measured and info["iter"] is not None:
            rate, span = measured
            source = f"measured {span / 60:.0f} min"
        else:
            rate = args.solo_rate
            source = "assumed solo"
        rates[arm] = rate
        if info["iter"] is None:
            print(f"{arm:<10} {'queued':>13}   {rate:.2f}*        {source:<17} waiting for a slot")
            continue
        remaining = max(0, args.budget - info["iter"])
        eta = now + remaining * rate
        print(f"{arm:<10} {info['iter']:>6}/{args.budget}   {rate:.2f}         {source:<17} "
              f"{time.strftime('%m-%d %H:%M', time.localtime(eta))}")
    print("-" * 74)

    # A queued arm starts when the *driver* frees a slot, and the driver only knows the arms it
    # launched itself -- a reference arm started separately can finish without releasing one.
    # That distinction matters here: baseline finishes about an hour before loss/aug, and reading
    # its completion as "a slot opened" would have pulled the last arm an hour early.
    managed = [arm for arm in args.arms if arm in set(args.driver_arms)]
    occupied: list[float] = []
    finish: dict[str, float] = {}
    for arm in managed:
        if state[arm]["done"]:
            continue
        if state[arm]["failed"]:
            continue
        if state[arm]["iter"] is None:
            continue
        end = now + max(0, args.budget - state[arm]["iter"]) * rates.get(arm, args.solo_rate)
        finish[arm] = end
        occupied.append(end)
    occupied.sort()

    for arm in args.arms:
        if arm not in managed or state[arm]["done"] or state[arm]["failed"]:
            continue
        if state[arm]["iter"] is not None:
            continue
        # Take the earliest slot that frees up, then hand that slot to the next queued arm.
        start = occupied.pop(0) if occupied else now
        duration = args.budget * args.solo_rate
        finish[arm] = start + duration
        occupied.append(finish[arm])
        occupied.sort()
        print(f"{arm:<10} projected start {time.strftime('%m-%d %H:%M', time.localtime(start))} "
              f"(modelled at {args.solo_rate:.2f} s/upd alone)")

    unmanaged = [arm for arm in args.arms
                 if state[arm]["iter"] is not None and not state[arm]["done"]
                 and not state[arm]["failed"] and arm not in managed]
    if unmanaged:
        print(f"\nnote: {', '.join(unmanaged)} was started outside the driver, so its completion "
              f"does not release a slot for a queued arm")

    if finish:
        last = max(finish.values())
        print(f"\nall driver arms done around {time.strftime('%m-%d %H:%M', time.localtime(last))} "
              f"(server time), i.e. {(last - now) / 3600:.1f} h from now")
        print("queued-arm timing is modelled, not measured; it will be replaced by a real rate "
              "once that arm starts")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
