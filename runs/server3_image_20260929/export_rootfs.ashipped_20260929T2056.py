"""Inspect or stream this container's root filesystem, excluding the competition dataset.

Runs **on the server**. `inspect` prints a JSON manifest describing what will be
excluded and the hashes of a set of critical files; `export` writes a
gzip-compressed tar stream of `/` to stdout.

The output is a *filesystem snapshot*: restore it with `docker import`, not
`docker load`, and never double-compress it (the stream is already gzip).

Usage:
    python export_rootfs.py inspect [--dataset DIR] [--critical FILE]...
    python export_rootfs.py export  [--dataset DIR]
"""

import argparse
import hashlib
import json
import re
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path

# Directories that must never enter a submission image.
DATASET_CANDIDATES = [
    "/root/hyperseg/dataset",
    "/root/dataset",
]

# Files whose presence and hash prove the archive is the machine we think it is.
DEFAULT_CRITICAL = [
    "/bin/bash",
    "/root/miniconda3/bin/python3.12",
    "/root/hyperseg/.venv-mask2former/pyvenv.cfg",
    "/root/hyperseg/hyperseg_uav/model.py",
    "/root/hyperseg/hyperseg_uav/__init__.py",
    "/root/hyperseg/tools/infer_hyperseg.py",
    "/root/hyperseg/tools/test_hyperseg.py",
    "/root/hyperseg/tools/check_submission.py",
    "/root/hyperseg/models/hyperseg_b3_best.pt",
    "/root/hyperseg/models/nvidia--mit-b3/config.json",
    "/root/hyperseg/runs/splits/train.txt",
    "/root/hyperseg/runs/splits/val.txt",
    "/root/hyperseg/runs/splits/test.txt",
]


def find_dataset():
    for candidate in DATASET_CANDIDATES:
        if Path(candidate).is_dir():
            return candidate
    return None


def exclusions(dataset):
    """Bind mounts plus the dataset. `--one-file-system` alone only stops tar from
    *descending* into another filesystem; files bind-mounted directly on top of the
    root filesystem still need naming here."""
    mounts = set()
    for line in Path("/proc/self/mountinfo").read_text().splitlines():
        path = re.sub(r"\\([0-7]{3})", lambda m: chr(int(m[1], 8)), line.split()[4])
        if path != "/":
            mounts.add(path)
    mounts = sorted(p for p in mounts if not any(
        p.startswith(parent + "/") for parent in mounts if parent != p))
    extra = [dataset] if dataset else []
    return sorted(set(mounts + extra + ["/.dockerenv", "/tmp/*", "/run/*"]))


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def inspect(dataset, critical):
    result = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "format": "gzip-compressed rootfs tar; use docker import, not docker load",
        "platform": "linux/amd64",
        "dataset_excluded": dataset,
        "excluded_paths": exclusions(dataset),
        "critical_files": {},
        "missing_critical": [],
        "os_release": Path("/etc/os-release").read_text(),
    }
    for name in critical:
        path = Path(name)
        if not path.is_file():
            result["missing_critical"].append(name)
            continue
        result["critical_files"][str(path.resolve())] = {
            "requested_path": name, "size": path.stat().st_size, "sha256": sha256(path)
        }
    print(json.dumps(result, indent=2), flush=True)


def export(dataset):
    with tempfile.TemporaryDirectory(prefix="rootfs-export-") as directory:
        exclusion_file = Path(directory) / "exclude.txt"
        exclusion_file.write_text("".join("." + name + "\n" for name in exclusions(dataset)))
        tar_command = [
            "tar", "--create", "--file=-", "--format=pax", "--numeric-owner",
            "--acls", "--xattrs", "--one-file-system", "--totals",
            "--exclude-from=" + str(exclusion_file), "--directory=/", ".",
        ]
        tar = subprocess.Popen(tar_command, stdout=subprocess.PIPE)
        # -3 keeps CPU cost low; the payload is mostly already-compressed bytes,
        # so a higher level buys almost nothing and doubles the export time.
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
    parser.add_argument("--dataset", default=None,
                        help="dataset directory to exclude (default: auto-detect)")
    parser.add_argument("--critical", action="append", default=None,
                        help="extra critical file to hash; replaces the default list when given")
    args = parser.parse_args()
    dataset = find_dataset() if args.dataset is None else args.dataset
    if dataset is None:
        # No dataset on this machine: exclude only mounts, but say so loudly.
        print("WARNING: no dataset directory found; nothing dataset-specific excluded",
              flush=True)
    if args.action == "inspect":
        inspect(dataset, args.critical or DEFAULT_CRITICAL)
    else:
        export(dataset)
