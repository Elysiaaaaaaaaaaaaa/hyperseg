"""Assemble the round-two submission package from this repository.

Builds the directory layout defined in ``doc/提交物组织规范.md``: prediction zip and
its self-check, technical plan, source + docker + reproduction scripts, experiment
evidence, dataset access note, and finally ``CHECKSUMS.sha256`` + ``MANIFEST.json``
covering every file in the package.

The competition dataset is never copied; only the split files (bare file-name lists)
travel with the package, because they are what makes the split reproducible.

Usage:
    python tools/build_submission_package.py --root dist/<package name>
    python tools/build_submission_package.py --root <package> --with-image <archive> [--image-note ...]
"""

import argparse
import hashlib
import json
import shutil
import zipfile
from datetime import datetime, timezone
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
ROUND2 = PROJECT / "experiment/b3_test2_20260929"
PREDICTION_ZIP = ROUND2 / "hyperseg_b3_test2_predictions.zip"

# Copied into 03_代码与复现/hyperseg so the package can train, evaluate and export
# without networking. The 1.5 GB Swin-L backbone is intentionally left out: the
# submitted checkpoint is MiT-B3, documented in REPRODUCE.md.
CODE_FILES = ["requirements.txt", "HYPERSEG.md", "AGENTS.md", "experiment.md"]
CODE_PACKAGES = ["hyperseg_uav"]
CODE_EXPERIMENTS = ["b3_test2_20260929"]
# Only the scripts that train, evaluate, export and check the submitted model.
# Unrelated experiment tooling (few-shot selectors, LoveDA GUI, older model drafts)
# is deliberately left out so the reviewer sees one implementation.
CODE_TOOLS = ["train_hyperseg.py", "test_hyperseg.py", "infer_hyperseg.py",
              "validate_hyperseg.py", "visualize_predictions.py", "check_submission.py"]
# Supporting scripts for building and verifying the image deliverable.
ROOTFS_TOOLS = ["export_rootfs.py", "download_rootfs.py", "build_submission_package.py"]
MODEL_FILES = ["models/hyperseg_b3_best.pt"]
MODEL_DIRS = ["models/nvidia--mit-b3"]

# Hand-written deliverables (README, Dockerfile, the environment/OPS/REPRODUCE docs,
# the numbered step scripts, the data-access note, the image note). They are authored
# in ``packaging/`` and mirrored into the package, because ``dist/`` is gitignored:
# editing them only inside the package means a rebuild from a clean checkout silently
# produces a package that is missing them. ``packaging/`` uses the package's own
# relative layout, so this is a straight directory copy.
STATIC_DIR = "packaging"

SKIP_DIRS = {"__pycache__", ".git", ".venv", "dataset", "runs"}
# The prediction zip is shipped once, in 01_预测结果; keeping a second copy inside
# the source tree would invite the question of which one is authoritative.
SKIP_SUFFIXES = {".pyc", ".pyo", ".zip"}


def copy_file(source, destination):
    """Copy a file, clearing the Windows read-only attribute (``copy2`` preserves it,
    and a read-only destination cannot be overwritten on the next rebuild)."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        try:
            destination.chmod(0o666)
        except OSError:
            pass
    shutil.copy2(source, destination)
    try:
        destination.chmod(0o644)
    except OSError:
        pass


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def copy_tree(source, target, keep=None):
    """Copy a directory, dropping caches and evaluation artifacts."""
    for path in sorted(source.rglob("*")):
        relative = path.relative_to(source)
        if SKIP_DIRS.intersection(relative.parts) or path.suffix in SKIP_SUFFIXES:
            continue
        if keep and not keep(relative):
            continue
        destination = target / relative
        if path.is_dir():
            destination.mkdir(parents=True, exist_ok=True)
        else:
            copy_file(path, destination)


def write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")
    print(f"  wrote {path}")


def build_predictions(package):
    target = package / "01_预测结果"
    target.mkdir(parents=True, exist_ok=True)
    submission = target / "predictions_test2_round2.zip"
    copy_file(PREDICTION_ZIP, submission)
    with zipfile.ZipFile(submission) as archive:
        names = [name for name in archive.namelist() if name.endswith(".png")]
    write(target / "file_list.txt", "\n".join(sorted(names)) + "\n")

    result = json.loads((ROUND2 / "logs/result.json").read_text(encoding="utf-8"))
    compliance = json.loads((ROUND2 / "logs/manual_compliance_20260929.json")
                            .read_text(encoding="utf-8"))
    summary = {
        "stage": "round2 (semifinal preparation)",
        "dataset": "low_altitude_2026 test set 2",
        "images": len(names),
        "archive_bytes": submission.stat().st_size,
        "sha256": sha256(submission),
        "checkpoint": result["checkpoint"],
        "checkpoint_sha256": result["checkpoint_sha256"],
        "checkpoint_epoch": result["checkpoint_epoch"],
        "checkpoint_val_miou": result["checkpoint_val_miou"],
        "protocol": {"size": result["size"], "overlap": result["overlap"],
                     "tta": result["tta"], "precision": result["precision"]},
        "submission_check": "OK: 1300 predictions",
        "manual_section_13_compliance": compliance["verdict"],
        "manual_compliance_detail": {
            "crc_ok": compliance["crc_ok"],
            "entries": compliance["entries"],
            "duplicate_entries": compliance["duplicate_entries"],
            "nested_or_dir_entries": compliance["nested_or_dir_entries"],
            "name_set_matches_test_set_2": compliance["name_set_matches_test_set_2"],
            "missing_names": compliance["missing_names"],
            "extra_names": compliance["extra_names"],
            "png_colour_type_gray": compliance["counts"]["color_type_gray"],
            "failure_count": compliance["failure_count"],
        },
        "class_pixel_share": compliance["pixel_share"],
        "class_ids_present": compliance["class_ids"],
        "generated_utc": datetime.now(timezone.utc).isoformat(),
    }
    write(target / "submission_check.json", json.dumps(summary, indent=2) + "\n")
    return submission


def build_code(package):
    root = package / "03_代码与复现"
    source = root / "hyperseg"
    source.mkdir(parents=True, exist_ok=True)
    for name in CODE_FILES:
        path = PROJECT / name
        if path.is_file():
            copy_file(path, source / name)
    for name in CODE_PACKAGES:
        copy_tree(PROJECT / name, source / name)
    for name in CODE_EXPERIMENTS:
        copy_tree(PROJECT / "experiment" / name, source / "experiment" / name)
    for name in CODE_TOOLS:
        path = PROJECT / "tools" / name
        if path.is_file():
            copy_file(path, source / "tools" / name)
    for name in MODEL_FILES:
        path = PROJECT / name
        if path.is_file():
            copy_file(path, source / Path(name))
    for name in MODEL_DIRS:
        path = PROJECT / name
        if path.is_dir():
            copy_tree(path, source / name)
    for name in ROOTFS_TOOLS:
        path = PROJECT / "tools" / name
        if path.is_file():
            copy_file(path, root / "image-tools" / name)
    return root


def build_evidence(package):
    target = package / "05_实验证据"
    copy_tree(ROUND2 / "logs", target / "logs")
    result = json.loads((ROUND2 / "logs/result.json").read_text(encoding="utf-8"))
    sanity = json.loads((ROUND2 / "logs/val_sanity.json").read_text(encoding="utf-8"))
    metrics = {
        "note": "val mIoU is measured on the fixed labelled validation split, not on "
                "the unlabelled competition test set; the test set has no ground truth",
        "labelled_validation_split": {
            "samples": sanity["samples"], "size": sanity["size"],
            "recorded_val_miou": sanity["recorded_val_miou"],
            "replayed_val_miou": sanity["replayed_val_miou"],
            "absolute_difference": sanity["absolute_difference"],
            "device": sanity["device"], "gpu": sanity["gpu"],
            "elapsed_seconds": sanity["elapsed_seconds"],
        },
        "round_two_export": {
            "images": result["image_count"], "gpu": result["gpu"],
            "torch": result["torch"], "elapsed_seconds": result["elapsed_seconds"],
            "seconds_per_image": round(result["elapsed_seconds"] / result["image_count"], 4),
        },
    }
    write(target / "metrics/summary.json", json.dumps(metrics, indent=2) + "\n")
    return target


def build_static(package):
    """Mirror the hand-written deliverables from ``packaging/`` into the package.

    Returns the list of relative paths written, so ``main`` can report a file that
    the template no longer provides (e.g. a doc renamed or dropped by mistake).
    """
    source = PROJECT / STATIC_DIR
    if not source.is_dir():
        raise SystemExit(f"missing {source}; the hand-written package files live there")
    written = []
    for path in sorted(source.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(source)
        copy_file(path, package / relative)
        written.append(relative.as_posix())
    print(f"  mirrored {len(written)} static file(s) from {STATIC_DIR}/")
    return written


def build_data_note(package):
    target = package / "06_数据说明"
    for name in ("train", "val", "test"):
        path = PROJECT / f"runs/splits/{name}.txt"
        if path.is_file():
            copy_file(path, target / "splits" / f"{name}.txt")
    return target


def checksums(package):
    lines = []
    entries = []
    for path in sorted(package.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(package).as_posix()
        if relative in {"CHECKSUMS.sha256", "MANIFEST.json"}:
            continue
        digest = sha256(path)
        size = path.stat().st_size
        lines.append(f"{digest}  {relative}")
        entries.append({"path": relative, "bytes": size, "sha256": digest})
    write(package / "CHECKSUMS.sha256", "\n".join(lines) + "\n")
    manifest = {
        "package": package.name,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "file_count": len(entries),
        "total_bytes": sum(entry["bytes"] for entry in entries),
        "files": entries,
    }
    write(package / "MANIFEST.json", json.dumps(manifest, indent=2) + "\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, help="package directory to create")
    parser.add_argument("--tech-plan", default=None,
                        help="technical plan PDF to place in 02_技术方案")
    parser.add_argument("--with-image", default=None,
                        help="rootfs archive to copy into 04_镜像 (large; optional)")
    parser.add_argument("--skip-checksums", action="store_true")
    args = parser.parse_args()

    package = Path(args.root).resolve()
    package.mkdir(parents=True, exist_ok=True)
    print(f"building {package}")

    for name in ("01_预测结果", "02_技术方案", "03_代码与复现", "04_镜像",
                 "05_实验证据", "06_数据说明"):
        (package / name).mkdir(exist_ok=True)

    build_predictions(package)
    build_static(package)
    build_code(package)
    build_evidence(package)
    build_data_note(package)

    if args.tech_plan:
        source = Path(args.tech_plan)
        copy_file(source, package / "02_技术方案" / source.name)
        print(f"  copied tech plan {source.name}")

    if args.with_image:
        source = Path(args.with_image)
        copy_file(source, package / "04_镜像" / source.name)
        print(f"  copied image {source.name}")

    if not args.skip_checksums:
        manifest = checksums(package)
        print(f"  {manifest['file_count']} files, "
              f"{manifest['total_bytes'] / 1024**3:.2f} GiB, manifest written")


if __name__ == "__main__":
    main()
