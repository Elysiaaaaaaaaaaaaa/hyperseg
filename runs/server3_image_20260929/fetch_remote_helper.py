"""Fetch the exact ``export_rootfs.py`` the paused archive was produced with.

The archive on disk was packed by whatever copy of the exporter was staged on the
source machine when the transfer started. That copy — not the current local file — is
what defines the byte stream a resume has to reproduce, so it is worth keeping a copy of
it next to the other transfer bookkeeping, and worth diffing against the local file
before deciding whether a resume or a fresh download is the honest option.

Also lists ``/root`` and the walk order of ``/`` so we can reason about where a partial
stream's cut-off lands.

Usage:
    python runs/server3_image_20260929/fetch_remote_helper.py <destination>
"""

import sys
import tempfile
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "tools"))

import download_rootfs as download  # noqa: E402


def main():
    destination = Path(sys.argv[1])
    bash = download.find_bash()
    config = download.parse_sshconfig("server3")
    with tempfile.TemporaryDirectory(prefix="fetch-helper-") as workdir:
        host = download.Host(config, workdir, bash)
        fetched = host.run("cat /root/export_rootfs.py", capture_output=True)
        listing = host.run("ls -la --time-style=full-iso /root | head -40; "
                           "echo '--- top level walk order ---'; ls -U /",
                           capture_output=True, text=True, encoding="utf-8",
                           errors="replace")
    if fetched.returncode:
        raise SystemExit(f"could not read the remote helper: {fetched.stderr[-800:]}")
    destination.write_bytes(fetched.stdout)
    print(f"wrote {destination} ({len(fetched.stdout)} bytes)")
    print(f"sha256={download.sha256_file(destination)}")
    local = PROJECT / "tools" / "export_rootfs.py"
    print(f"local tools/export_rootfs.py ({local.stat().st_size} bytes) "
          f"sha256={download.sha256_file(local)}")
    print("--- remote /root ---")
    print(listing.stdout)


if __name__ == "__main__":
    main()
