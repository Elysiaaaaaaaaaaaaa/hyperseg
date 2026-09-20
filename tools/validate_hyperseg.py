"""Evaluate a HyperSeg-UAV checkpoint on a labeled split.

The metric is computed from one dataset-wide confusion matrix.  Ignore pixels
(label 0) are excluded from both the intersection and the union, and the
reported mIoU is the mean IoU of classes 1..8 that occur in the split.
"""

import argparse
import json
import sys
from io import TextIOWrapper
from pathlib import Path
from zipfile import ZipFile

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hyperseg_uav import HyperSegUAV, UAVDataset


CLASS_NAMES = {
    0: "Ignore",
    1: "Background",
    2: "Building",
    3: "Road",
    4: "Water",
    5: "Barren",
    6: "Vegetation",
    7: "Agricultural",
    8: "Vehicle",
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT / "models/hyperseg_resume_best.pt",
        help="checkpoint containing model and model_config",
    )
    parser.add_argument("--data", type=Path, default=ROOT / "dataset")
    parser.add_argument(
        "--split-dir",
        type=Path,
        default=ROOT / "runs/splits",
        help="directory containing train.txt, val.txt and test.txt",
    )
    parser.add_argument(
        "--split-archive",
        type=Path,
        default=ROOT / "splits.zip",
        help="fallback ZIP archive containing splits/<split>.txt",
    )
    parser.add_argument(
        "--split",
        choices=("train", "val", "test"),
        default="val",
        help="split to evaluate (default: val)",
    )
    parser.add_argument("--size", type=int, default=768)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "cpu"),
        default="auto",
        help="inference device; auto selects CUDA when available",
    )
    parser.add_argument(
        "--metrics-out",
        type=Path,
        default=None,
        help="optional JSON output path (default: next to checkpoint)",
    )
    return parser.parse_args()


def _parse_ids(text):
    ids = [line.strip() for line in text.splitlines() if line.strip()]
    if not ids:
        raise ValueError("split file is empty")
    return ids


def read_ids(split_dir, split, split_archive=None):
    split_path = split_dir / f"{split}.txt"
    if split_path.is_file():
        try:
            ids = _parse_ids(split_path.read_text(encoding="utf-8"))
        except ValueError as error:
            raise ValueError(f"split file is empty: {split_path}") from error
        return ids, str(split_path)

    if split_archive and split_archive.is_file():
        with ZipFile(split_archive) as archive:
            candidates = [
                name for name in archive.namelist()
                if name.endswith(f"/{split}.txt") or name == f"{split}.txt"
            ]
            if candidates:
                member = candidates[0]
                with TextIOWrapper(archive.open(member), encoding="utf-8") as handle:
                    try:
                        ids = _parse_ids(handle.read())
                    except ValueError as error:
                        raise ValueError(f"split file is empty: {split_archive}!{member}") from error
                return ids, f"{split_archive}!{member}"

    raise FileNotFoundError(
        f"split file not found: {split_path}; archive not found or missing {split}.txt: "
        f"{split_archive}"
    )


def select_device(requested):
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested, but CUDA is not available")
    return torch.device(requested)


def build_model(checkpoint, device):
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    if not isinstance(state, dict) or "model" not in state:
        raise ValueError(f"invalid checkpoint format: {checkpoint}")
    config = dict(state.get("model_config", {}))
    # The checkpoint contains trained weights; loading a second pretrained
    # backbone would be unnecessary and could require network access.
    config["pretrained"] = False
    model = HyperSegUAV(**config).to(device)
    incompatible = model.load_state_dict(state["model"], strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            "checkpoint does not match the reconstructed model: "
            f"missing={incompatible.missing_keys}, "
            f"unexpected={incompatible.unexpected_keys}"
        )
    model.eval()
    return model, state


@torch.no_grad()
def evaluate(model, loader, device, classes=9, ignore_index=0):
    confusion = torch.zeros(classes, classes, dtype=torch.int64, device=device)
    sample_count = 0
    ignored_pixels = 0
    valid_pixels = 0

    for batch in loader:
        image = batch["image"].to(device, non_blocking=True)
        target = batch["mask"].to(device, non_blocking=True)
        prediction = model(image)["logits"].argmax(dim=1)

        valid = (target != ignore_index) & (target >= 0) & (target < classes)
        ignored_pixels += int((target == ignore_index).sum())
        valid_pixels += int(valid.sum())
        indices = target[valid] * classes + prediction[valid]
        confusion += torch.bincount(
            indices, minlength=classes * classes
        ).reshape(classes, classes)
        sample_count += image.shape[0]

    intersection = confusion.diag().float()
    union = confusion.sum(0).float() + confusion.sum(1).float() - intersection
    present = union[1:] > 0
    per_class = {}
    for cls in range(1, classes):
        per_class[str(cls)] = {
            "name": CLASS_NAMES.get(cls, f"class_{cls}"),
            "iou": float(intersection[cls] / union[cls]) if union[cls] else 0.0,
            "present": bool(present[cls - 1]),
        }
    miou = float(
        (intersection[1:][present] / union[1:][present]).mean()
    ) if present.any() else 0.0
    return {
        "samples": sample_count,
        "valid_pixels": valid_pixels,
        "ignored_pixels": ignored_pixels,
        "mIoU": miou,
        "per_class_iou": per_class,
        "confusion": confusion.cpu().tolist(),
    }


def main():
    args = parse_args()
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint not found: {args.checkpoint}")

    ids, split_path = read_ids(args.split_dir, args.split, args.split_archive)
    image_dir = args.data / "train" / "train" / "images"
    mask_dir = args.data / "train" / "train" / "masks"
    if not image_dir.is_dir():
        raise FileNotFoundError(f"image directory not found: {image_dir}")
    if not mask_dir.is_dir():
        raise FileNotFoundError(f"mask directory not found: {mask_dir}")

    device = select_device(args.device)
    dataset = UAVDataset(image_dir, mask_dir, ids, args.size, training=False)
    if len(dataset) != len(ids):
        raise ValueError(
            f"split contains {len(ids)} IDs, but only {len(dataset)} matching images "
            f"were found in {image_dir}"
        )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    model, checkpoint_state = build_model(args.checkpoint, device)
    metrics = evaluate(model, loader, device)
    metrics.update(
        {
            "checkpoint": str(args.checkpoint),
            "checkpoint_epoch": checkpoint_state.get("epoch"),
            "checkpoint_val_miou": checkpoint_state.get("val_miou"),
            "split": args.split,
            "split_file": str(split_path),
            "device": str(device),
            "size": args.size,
            "classes": 9,
            "ignore_index": 0,
        }
    )

    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    output_path = args.metrics_out or args.checkpoint.with_name(
        f"{args.checkpoint.stem}_{args.split}_metrics.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"metrics written to: {output_path}")


if __name__ == "__main__":
    main()
