"""Move the scratch tooling left in ``/root`` on the source machine out of the export.

Anything under ``/root`` ends up inside the submitted image, so the helper scripts and
probe outputs an interrupted transfer leaves behind must not stay there. They are *moved*
into the data disk (``/root/autodl-tmp``), which ``export_rootfs.py`` excludes, rather
than deleted, so nothing is lost.

Usage:
    python runs/server3_image_20260929/clean_remote_scratch.py [--apply]
"""

import argparse
import sys
import tempfile
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "tools"))

import download_rootfs as download  # noqa: E402

STAGING = "/root/autodl-tmp/scratch-20260929"
SCRATCH = [
    "/root/export_rootfs.py",
    "/root/probe_export_stream.py",
    "/root/prefix_probe.json",
    "/root/prefix_probe.err",
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", default="server3")
    parser.add_argument("--apply", action="store_true",
                        help="actually move the files; without it, only report")
    args = parser.parse_args()

    if args.apply:
        commands = [f"mkdir -p {STAGING}"] + [
            f"[ -e {path} ] && mv -v {path} {STAGING}/" for path in SCRATCH]
    else:
        commands = [f"[ -e {path} ] && echo would move {path}" for path in SCRATCH]
    commands.append("echo '--- /root now ---'; ls -la /root")
    commands.append("echo '--- staged ---'; ls -la " + STAGING + " 2>/dev/null")

    bash = download.find_bash()
    config = download.parse_sshconfig(args.server)
    with tempfile.TemporaryDirectory(prefix="clean-scratch-") as workdir:
        host = download.Host(config, workdir, bash)
        result = host.run("; ".join(commands), capture_output=True, text=True,
                          encoding="utf-8", errors="replace")
    print(f"exit={result.returncode}\n{result.stdout}")
    if result.stderr.strip():
        print(f"--- stderr ---\n{result.stderr[-800:]}")


if __name__ == "__main__":
    main()
