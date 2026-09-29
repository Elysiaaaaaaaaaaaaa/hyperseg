"""Finish the active transfer, then defer verification to the user's next step."""
import json
import os
import signal
import sys
import time
from pathlib import Path

from download_server2_rootfs import ARCHIVE, MANIFEST, PARTIAL, STATUS, update


def main(pid):
    command = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
    if b"tools/download_server2_rootfs.py" not in command:
        raise RuntimeError("PID does not match the active download process")
    # Verify the process start time again before signalling a PID.
    proc_stat = Path(f"/proc/{pid}/stat")
    started = proc_stat.read_text().rsplit(")", 1)[1].split()[19]
    Path(str(STATUS) + ".deferred").write_text(
        json.dumps({"download_pid": pid, "watcher_pid": os.getpid(),
                    "verification": "deferred until manually requested"}) + "\n")
    try:
        while True:
            state = json.loads(STATUS.read_text())
            stage = state.get("stage")
            if stage == "verifying":
                # The downloader enters this stage only after EOF, SSH/tar exit 0,
                # destination flush/fsync, and closing the output file.
                if proc_stat.read_text().rsplit(")", 1)[1].split()[19] != started:
                    raise RuntimeError("Download PID was reused")
                os.kill(pid, signal.SIGTERM)
                time.sleep(0.5)
                if not PARTIAL.is_file() or ARCHIVE.exists():
                    raise RuntimeError("Unexpected output state; keep files for inspection")
                PARTIAL.replace(ARCHIVE)
                manifest = json.loads(MANIFEST.read_text())
                manifest.update(archive=ARCHIVE.name, compressed_bytes=ARCHIVE.stat().st_size,
                                verification="pending_user_request")
                MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")
                update(stage="downloaded_unverified", archive=ARCHIVE.name,
                       bytes=ARCHIVE.stat().st_size, verification="pending_user_request",
                       transfer_exit_code=0)
                return
            if stage in {"complete", "failed", "downloaded_unverified"}:
                print(json.dumps(state), flush=True)
                return
            if not Path(f"/proc/{pid}").exists():
                raise RuntimeError("Download process exited before transfer completion")
            time.sleep(0.1)
    finally:
        print("Background transfer watcher finished", flush=True)


if __name__ == "__main__":
    main(int(sys.argv[1]))
