"""Run on server2: inspect or stream its system filesystem without datasets."""
import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

DATASET = "/root/hyperseg/dataset"
CRITICAL = [
    "/bin/bash",
    "/root/miniconda3/bin/python3.12",
    "/root/hyperseg/.venv-mask2former/pyvenv.cfg",
    "/root/hyperseg/hyperseg_uav/swin_l_model.py",
    "/root/hyperseg/runs/mask2former_uav/h3_swin_l_hyperseg_160k_seed3407_fp32/best.pt",
    "/root/hyperseg/runs/splits/train.txt",
    "/root/hyperseg/runs/splits/val.txt",
    "/root/hyperseg/runs/splits/test.txt",
]


def exclusions():
    # Bind-mounted files are excluded explicitly as --one-file-system alone
    # only stops descent into directories on another filesystem.
    mounts = set()
    for line in Path("/proc/self/mountinfo").read_text().splitlines():
        path = re.sub(r"\\([0-7]{3})", lambda m: chr(int(m[1], 8)), line.split()[4])
        if path != "/":
            mounts.add(path)
    mounts = sorted(p for p in mounts if not any(
        p.startswith(parent + "/") for parent in mounts if parent != p))
    return sorted(set(mounts + [DATASET, "/.dockerenv", "/tmp/*", "/run/*"]))


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def inspect():
    result = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "format": "gzip-compressed rootfs tar; use docker import, not docker load",
        "platform": "linux/amd64",
        "excluded_paths": exclusions(),
        "critical_files": {},
        "os_release": Path("/etc/os-release").read_text(),
    }
    for name in CRITICAL:
        path = Path(name)
        result["critical_files"][str(path.resolve())] = {
            "requested_path": name, "size": path.stat().st_size, "sha256": sha256(path)
        }
    print(json.dumps(result, indent=2), flush=True)


def export():
    with tempfile.TemporaryDirectory(prefix="server2-rootfs-export-") as directory:
        exclusion_file = Path(directory) / "exclude.txt"
        exclusion_file.write_text("".join("." + name + "\n" for name in exclusions()))
        tar_command = [
            "tar", "--create", "--file=-", "--format=pax", "--numeric-owner",
            "--acls", "--xattrs", "--one-file-system", "--totals",
            "--exclude-from=" + str(exclusion_file), "--directory=/", ".",
        ]
        tar = subprocess.Popen(tar_command, stdout=subprocess.PIPE)
        compressor = subprocess.Popen(["gzip", "-3", "-n"], stdin=tar.stdout)
        tar.stdout.close()
        try:
            gzip_status = compressor.wait()
            tar_status = tar.wait()
        finally:
            for process in (tar, compressor):
                if process.poll() is None:
                    process.terminate()
                    process.wait()
        if gzip_status or tar_status:
            raise SystemExit(f"Export failed: tar={tar_status}, gzip={gzip_status}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["inspect", "export"])
    args = parser.parse_args()
    if args.action == "inspect":
        inspect()
    else:
        export()
