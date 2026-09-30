"""Watch the H3 bgfix screening arms and report progress until they all finish.

    python experiment/h3_bgfix_20260930/watch_screen.py --interval 300

Polls each arm's ``exit_code`` on the server and prints one progress line per arm until every
arm has finished, then hands over to ``summarize_screen.py``.  The training itself is detached
on the server, so this script only observes -- killing it loses nothing.  ``--once`` prints a
single snapshot and returns.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import remote  # noqa: E402


def snapshot(client, arms, suffix):
    """Return ``{arm: (status, latest_iter, validations)}`` where status is done/running/failed.

    The fields are emitted as ``KEY=value`` lines rather than bare lines: an absent
    ``exit_code`` produces no output at all, so anything positional would silently shift the
    remaining fields up and report a healthy arm as failed.
    """
    out = {}
    for arm in arms:
        work_dir = f"/root/hyperseg/runs/mask2former_uav/h3_bgfix_{arm}_{suffix}_seed3407"
        _, text = remote.run(
            client,
            f'echo "EXIT=$(cat {work_dir}/exit_code 2>/dev/null | head -n1)"; '
            f'echo "COUNT=$(wc -l < {work_dir}/val_history.jsonl 2>/dev/null | tr -d " ")"; '
            f'echo "ITER=$(grep -o \'^iter [0-9]*\' {work_dir}/train.log 2>/dev/null | tail -n1)"',
        )
        fields = {}
        for line in text.splitlines():
            key, separator, value = line.partition("=")
            if separator:
                fields[key.strip()] = value.strip()
        exit_code = fields.get("EXIT", "")
        last_iter = fields.get("ITER", "")
        count = fields.get("COUNT", "")
        if exit_code:
            status = "done" if exit_code == "0" else f"failed({exit_code})"
        else:
            status = "running" if (last_iter or count) else "not started"
        out[arm] = (status, last_iter, count)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--server", default="server2", choices=("server1", "server2", "server3"))
    parser.add_argument("--suffix", default="20k")
    parser.add_argument("--arms", nargs="+", default=["baseline", "loss", "aug", "full"])
    parser.add_argument("--interval", type=int, default=300, help="seconds between polls")
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()

    while True:
        client = remote.connect(args.server)
        try:
            state = snapshot(client, args.arms, args.suffix)
        finally:
            client.close()

        stamp = time.strftime("%Y-%m-%d %H:%M:%S")
        summary = "  ".join(f"{arm}={status.split('(')[0]}[{last_iter or '-'}|v{count}]"
                            for arm, (status, last_iter, count) in state.items())
        print(f"{stamp}  {summary}", flush=True)

        failed = [arm for arm, (status, _, _) in state.items() if status.startswith("failed")]
        if failed:
            print(f"\nABORT: {', '.join(failed)} failed; inspect their train.log", flush=True)
            return 1
        if all(status == "done" for status, _, _ in state.values()):
            print("\nall arms finished; summarising\n", flush=True)
            return subprocess.call([sys.executable, str(HERE / "summarize_screen.py"),
                                    "--server", args.server, "--suffix", args.suffix,
                                    *sum((["--arms"], args.arms), [])])
        if args.once:
            return 0
        time.sleep(args.interval)


if __name__ == "__main__":
    raise SystemExit(main())
