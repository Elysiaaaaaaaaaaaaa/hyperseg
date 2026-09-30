"""One-shot read-only look at the source machine, for planning a resume.

Answers the three questions that decide whether an interrupted transfer can be picked up:

* is the host reachable at all;
* what does ``/root/export_rootfs.py`` look like now — size and, crucially, mtime, since
  the archive already on disk recorded that mtime and a regenerated stream has to
  reproduce it;
* is the data disk the stream helper is staged on really there and really excluded.

Usage:
    python runs/server3_image_20260929/probe_remote_state.py
"""

import sys
import tempfile
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "tools"))

import download_rootfs as download  # noqa: E402

COMMAND = (
    "hostname; "
    "echo '--- uname ---'; uname -sr; "
    "echo '--- disk ---'; df -h / | tail -1; "
    "echo '--- data disk ---'; df -h /root/autodl-tmp 2>/dev/null | tail -1 "
    "|| echo 'MISSING /root/autodl-tmp'; "
    "echo '--- helper ---'; stat -c '%n size=%s mtime=%Y %y' /root/export_rootfs.py 2>&1; "
    "sha256sum /root/export_rootfs.py 2>&1; "
    "echo '--- stray helper in root ---'; "
    "stat -c '%n mtime=%Y' /root/remote_export_stream.py 2>&1; "
    "echo '--- root mtime ---'; stat -c '%n mtime=%Y %y' /root; "
    "echo '--- logs ---'; stat -c '%n size=%s mtime=%Y' /var/log/wtmp /var/log/lastlog "
    "2>&1; "
    "echo '--- uptime ---'; uptime"
)


def main():
    bash = download.find_bash()
    config = download.parse_sshconfig("server3")
    with tempfile.TemporaryDirectory(prefix="remote-probe-") as workdir:
        host = download.Host(config, workdir, bash)
        result = host.run(COMMAND, capture_output=True, text=True,
                          encoding="utf-8", errors="replace")
    print(f"exit={result.returncode}")
    print("--- stdout ---")
    print(result.stdout)
    if result.stderr.strip():
        print("--- stderr ---")
        print(result.stderr[-1500:])
    print("--- local helper ---")
    local = PROJECT / "tools" / "export_rootfs.py"
    print(f"size={local.stat().st_size} sha256={download.sha256_file(local)}")


if __name__ == "__main__":
    main()
