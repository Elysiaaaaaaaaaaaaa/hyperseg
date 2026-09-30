"""Assemble the full delivery tree: the submission package plus the image volumes.

``tools/build_submission_package.py`` builds the lightweight package (code, plan,
predictions, evidence, image *note*).  The image itself -- 12.18 GiB split into four
volumes so each file fits under a netdisk's per-file cap -- lives outside it, because
carrying it inside the package would make the package unusable for any upload endpoint
with a size limit.

The competition asks for one deliverable though, so this script merges the two into a
single tree that can be uploaded as-is:

    <delivery>/                      = a copy of the package
    <delivery>/04_镜像/part_01…04     = the image volumes, copied in
    <delivery>/CHECKSUMS.sha256      = recomputed over *everything*, volumes included
    <delivery>/MANIFEST.json         = same, machine readable

The volumes are copied, never moved: on the exFAT delivery drive a rename of an
existing file is refused outright, and a copy can be verified while the source stays
put as a backup.

``04_镜像/CHECKSUMS.sha256`` keeps its own volume-only list, so the image directory is
still self-describing when it is separated again.

Usage:
    python tools/assemble_delivery_tree.py \
        --package "dist/<package name>" \
        --volumes "<dir with the part_* files>" \
        --out "E:/path/to/<delivery name>"
"""

import argparse
import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

# A netdisk's per-file ceiling is what forces the split; a volume that is too large is
# worse than one extra volume, so the default stays comfortably under 4 GB.
VOLUME_GLOB = "hyperseg-rootfs-*.tar.gz.part_*"
IMAGE_DIR = "04_镜像"


def sha256_file(path, block=8 * 1024 * 1024):
    digest = hashlib.sha256()
    total = 0
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(block)
            if not chunk:
                break
            digest.update(chunk)
            total += len(chunk)
    return digest.hexdigest(), total


def copy_tree(source, destination):
    """Copy a directory tree, overwriting files that are already there.

    ``copytree`` without ``dirs_exist_ok`` refuses a non-empty destination, and a
    half-built delivery directory is exactly the case this has to survive.
    """
    destination.mkdir(parents=True, exist_ok=True)
    copied = 0
    for path in sorted(source.rglob("*")):
        relative = path.relative_to(source)
        target = destination / relative
        if path.is_dir():
            target.mkdir(parents=True, exist_ok=True)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            try:
                target.chmod(0o666)
            except OSError:
                pass
        shutil.copy2(path, target)
        copied += 1
    return copied


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", required=True,
                        help="package directory produced by build_submission_package.py")
    parser.add_argument("--volumes", required=True,
                        help="directory holding the hyperseg-rootfs-*.tar.gz.part_* volumes")
    parser.add_argument("--out", required=True, help="delivery directory to create")
    parser.add_argument("--volume-glob", default=VOLUME_GLOB)
    args = parser.parse_args()

    package = Path(args.package).resolve()
    volume_dir = Path(args.volumes).resolve()
    out = Path(args.out)

    if not package.is_dir():
        raise SystemExit(f"no package at {package}")
    volumes = sorted(volume_dir.glob(args.volume_glob))
    if not volumes:
        raise SystemExit(f"no volumes matching {args.volume_glob!r} under {volume_dir}")

    print(f"package : {package}")
    print(f"volumes : {len(volumes)} file(s) from {volume_dir}")
    for volume in volumes:
        print(f"          {volume.name}  {volume.stat().st_size:,} B")
    print(f"out     : {out}")

    print("\n[1/4] copying the package")
    print(f"  {copy_tree(package, out)} file(s) copied")

    print("\n[2/4] copying the image volumes into " + IMAGE_DIR)
    target_dir = out / IMAGE_DIR
    target_dir.mkdir(parents=True, exist_ok=True)
    for volume in volumes:
        target = target_dir / volume.name
        if target.exists():
            try:
                target.chmod(0o666)
            except OSError:
                pass
        shutil.copy2(volume, target)
        print(f"  {volume.name}  {target.stat().st_size:,} B")

    # The volume-only checksum list travels with the volumes: it is written relative to
    # the image directory, so the image stays self-describing when it is split off
    # again. It is *generated* rather than copied from the source directory, so it can
    # only ever describe the bytes that actually landed in the delivery tree -- a stale
    # hand-maintained copy is exactly how a wrong volume hash survives undetected.

    print("\n[3/4] hashing every file in the delivery tree")
    entries = []
    for path in sorted(out.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(out).as_posix()
        # Never let a checksum list hash itself. Matching on the bare file name rather
        # than the root-relative path also keeps ``04_镜像/CHECKSUMS.sha256`` -- written
        # just below -- out of the tree-wide list, so the two stay independent and the
        # script is idempotent on a directory it has already built.
        if path.name in ("CHECKSUMS.sha256", "MANIFEST.json"):
            continue
        digest, size = sha256_file(path)
        entries.append({"path": relative, "bytes": size, "sha256": digest})
        print(f"  {digest[:16]}…  {size:>13,}  {relative}")

    checksum_path = out / "CHECKSUMS.sha256"
    checksum_path.write_text(
        "".join(f"{entry['sha256']}  {entry['path']}\n" for entry in entries),
        encoding="utf-8",
        # Without this, Windows rewrites every "\n" as "\r\n" and `sha256sum -c` fails
        # with "No such file or directory" on names that carry a trailing carriage
        # return. The list has to be byte-identical to what the tooling on Linux expects.
        newline="\n",
    )
    print(f"\n  wrote {checksum_path.name} ({len(entries)} entries)")

    manifest = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "file_count": len(entries),
        "total_bytes": sum(entry["bytes"] for entry in entries),
        "files": entries,
    }
    (out / "MANIFEST.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    print(f"  wrote MANIFEST.json  ({manifest['file_count']} files, "
          f"{manifest['total_bytes'] / 1024**3:.2f} GiB)")

    # Self-contained list for the image directory: the same bytes, with paths relative
    # to it. Written from the hashes computed above, so it describes what is actually
    # on disk rather than what some earlier step believed.
    image_entries = [entry for entry in entries
                     if entry["path"].startswith(IMAGE_DIR + "/")]
    if image_entries:
        prefix_length = len(IMAGE_DIR) + 1
        (target_dir / "CHECKSUMS.sha256").write_text(
            "".join(f"{entry['sha256']}  {entry['path'][prefix_length:]}\n"
                    for entry in image_entries),
            encoding="utf-8",
            newline="\n",
        )
        print(f"  wrote {IMAGE_DIR}/CHECKSUMS.sha256 ({len(image_entries)} entries)")

    print("\n[4/4] result")
    print(f"  files      : {manifest['file_count']}")
    print(f"  total      : {manifest['total_bytes']:,} B "
          f"({manifest['total_bytes'] / 1024**3:.2f} GiB)")
    volume_bytes = sum(entry["bytes"] for entry in entries
                       if entry["path"].startswith(IMAGE_DIR + "/"))
    print(f"  of which   : {volume_bytes:,} B of image volumes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
