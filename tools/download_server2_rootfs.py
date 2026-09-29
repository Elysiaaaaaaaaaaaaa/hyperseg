"""Stream the server2 rootfs backup to the project root, then verify it."""
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "server2-system-no-dataset-20260926"
ARCHIVE = Path(str(BASE) + ".rootfs.tar.gz")
PARTIAL = Path(str(ARCHIVE) + ".partial")
MANIFEST = Path(str(BASE) + ".manifest.json")
STATUS = Path(str(BASE) + ".status.json")
REMOTE = "/tmp/export_server2_rootfs_20260926.py"


def update(**values):
    values["updated_utc"] = datetime.now(timezone.utc).isoformat()
    temp = Path(str(STATUS) + ".tmp")
    temp.write_text(json.dumps(values, indent=2) + "\n")
    temp.replace(STATUS)
    print(json.dumps(values), flush=True)


def remote(action, **kwargs):
    return subprocess.Popen(
        [sys.executable, str(ROOT / ".tmp_server2_access.py"), "exec",
         f"/root/miniconda3/bin/python {REMOTE} {action}"], cwd=ROOT, **kwargs,
    )


def verify(path, manifest):
    critical = manifest["critical_files"]
    found = {}
    total_bytes = members = 0
    forbidden = [p.lstrip("/") for p in manifest["excluded_paths"] if "*" not in p]
    update(stage="verifying", archive_bytes=path.stat().st_size)
    last_update = time.monotonic()
    with tarfile.open(path, mode="r|gz") as archive:
        for member in archive:
            name = member.name.removeprefix("./").rstrip("/")
            if name.startswith("/") or ".." in name.split("/"):
                raise RuntimeError(f"Unexpected archive path: {name}")
            if any(name == p or name.startswith(p + "/") for p in forbidden):
                raise RuntimeError(f"Excluded path found in archive: {name}")
            if name.startswith(("tmp/", "run/")):
                raise RuntimeError(f"Transient content found in archive: {name}")
            members += 1
            if member.isfile():
                total_bytes += member.size
            key = "/" + name
            if key in critical:
                if not member.isfile():
                    raise RuntimeError(f"Critical file is not regular: {name}")
                digest = hashlib.sha256()
                with archive.extractfile(member) as stream:
                    for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                        digest.update(block)
                expected = critical[key]
                if member.size != expected["size"] or digest.hexdigest() != expected["sha256"]:
                    raise RuntimeError(f"Critical file differs from source: {name}")
                found[key] = digest.hexdigest()
            if time.monotonic() - last_update > 30:
                update(stage="verifying", members=members, file_bytes=total_bytes)
                last_update = time.monotonic()
    missing = set(critical) - set(found)
    if missing:
        raise RuntimeError(f"Missing critical files: {sorted(missing)}")
    # gzip -t consumes the complete stream, including the trailer after tar EOF.
    subprocess.run(["gzip", "-t", str(path)], check=True)
    return {"members": members, "regular_file_bytes": total_bytes,
            "critical_files_verified": found, "excluded_paths_verified": True,
            "gzip_integrity_verified": True}


def main():
    os.chdir(ROOT)
    if ARCHIVE.exists() or PARTIAL.exists():
        raise RuntimeError("Archive or partial file already exists; refusing to overwrite")
    if shutil.disk_usage(ROOT).free < 24 * 1024**3:
        raise RuntimeError("Need 24 GiB free before starting this full system export")
    update(stage="source_manifest")
    probe = remote("inspect", stdout=subprocess.PIPE)
    output, _ = probe.communicate()
    if probe.returncode:
        raise RuntimeError(f"Source inspection failed: {probe.returncode}")
    manifest = json.loads(output)
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")
    digest = hashlib.sha256()
    size = 0
    start = last_update = time.monotonic()
    update(stage="downloading", bytes=0)
    with PARTIAL.open("xb") as destination:
        transfer = remote("export", stdout=subprocess.PIPE)
        try:
            while block := transfer.stdout.read(4 * 1024 * 1024):
                if shutil.disk_usage(ROOT).free < len(block) + 2 * 1024**3:
                    raise RuntimeError("Stopping before local disk free space falls below 2 GiB")
                destination.write(block)
                digest.update(block)
                size += len(block)
                now = time.monotonic()
                if now - last_update > 30:
                    destination.flush()
                    update(stage="downloading", bytes=size, elapsed_seconds=round(now-start),
                           average_mib_s=round(size / (now-start) / 1024**2, 2))
                    last_update = now
            if transfer.wait():
                raise RuntimeError(f"Remote tar/SSH failed: {transfer.returncode}; partial retained")
            destination.flush()
            os.fsync(destination.fileno())
        finally:
            if transfer.poll() is None:
                transfer.terminate()
                transfer.wait()
    verification = verify(PARTIAL, manifest)
    PARTIAL.replace(ARCHIVE)
    Path(str(ARCHIVE) + ".sha256").write_text(f"{digest.hexdigest()}  {ARCHIVE.name}\n")
    manifest.update(archive=ARCHIVE.name, compressed_bytes=size, sha256=digest.hexdigest(),
                    verification=verification)
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")
    update(stage="complete", archive=ARCHIVE.name, bytes=size, sha256=digest.hexdigest(),
           elapsed_seconds=round(time.monotonic()-start))


def verify_existing():
    state = json.loads(STATUS.read_text())
    if state.get("stage") not in {"downloaded_unverified", "complete", "verification_failed"}:
        raise RuntimeError("Wait for stage=downloaded_unverified before manual verification")
    manifest = json.loads(MANIFEST.read_text())
    verification = verify(ARCHIVE, manifest)
    digest = hashlib.sha256()
    with ARCHIVE.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    Path(str(ARCHIVE) + ".sha256").write_text(f"{digest.hexdigest()}  {ARCHIVE.name}\n")
    manifest.update(archive=ARCHIVE.name, compressed_bytes=ARCHIVE.stat().st_size,
                    sha256=digest.hexdigest(), verification=verification)
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")
    update(stage="complete", archive=ARCHIVE.name, bytes=ARCHIVE.stat().st_size,
           sha256=digest.hexdigest(), verification_requested="manual")


if __name__ == "__main__":
    try:
        if sys.argv[1:] == ["--verify-existing"]:
            verify_existing()
        else:
            main()
    except BaseException as exc:
        update(stage="verification_failed" if "--verify-existing" in sys.argv else "failed", error=str(exc))
        raise
