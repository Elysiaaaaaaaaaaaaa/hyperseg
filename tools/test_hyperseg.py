"""Evaluate a HyperSeg-UAV checkpoint on the fixed test split."""
import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hyperseg_uav import HyperSegUAV, UAVDataset


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=ROOT / "dataset")
    parser.add_argument("--split-dir", type=Path, default=ROOT / "runs/splits")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "runs/hyperseg_b3_best.pt")
    parser.add_argument("--size", type=int, default=768)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=2)
    return parser.parse_args()


@torch.no_grad()
def evaluate(model, loader, device, classes=9, ignore_index=0):
    confusion = torch.zeros(classes, classes, dtype=torch.int64, device=device)
    model.eval()
    for batch in loader:
        image = batch["image"].to(device, non_blocking=True)
        target = batch["mask"].to(device, non_blocking=True)
        prediction = model(image)["logits"].argmax(dim=1)
        valid = (target != ignore_index) & (target >= 0) & (target < classes)
        indices = target[valid] * classes + prediction[valid]
        confusion += torch.bincount(indices, minlength=classes * classes).reshape(classes, classes)

    intersection = confusion.diag().float()
    union = confusion.sum(0).float() + confusion.sum(1).float() - intersection
    per_class = {
        cls: float(intersection[cls] / union[cls]) if union[cls] > 0 else 0.0
        for cls in range(1, classes)
    }
    valid = union[1:] > 0
    miou = float((intersection[1:][valid] / union[1:][valid].clamp_min(1)).mean()) if valid.any() else 0.0
    return miou, per_class, confusion.cpu().tolist()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_file = args.split_dir / "test.txt"
    test_ids = [line.strip() for line in test_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    image_dir = args.data / "train" / "train" / "images"
    mask_dir = args.data / "train" / "train" / "masks"
    dataset = UAVDataset(image_dir, mask_dir, test_ids, args.size, training=False)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=device.type == "cuda")
    state = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = dict(state.get("model_config", {}))
    config["pretrained"] = False
    model = HyperSegUAV(**config).to(device)
    incompatible = model.load_state_dict(state["model"], strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        print(f"checkpoint compatibility: missing={len(incompatible.missing_keys)}, unexpected={len(incompatible.unexpected_keys)}")
    miou, per_class, confusion = evaluate(model, loader, device)
    result = {"checkpoint": str(args.checkpoint), "samples": len(dataset), "mIoU": miou, "per_class_iou": per_class, "confusion": confusion}
    print(json.dumps(result, indent=2))
    result_path = args.checkpoint.with_name(args.checkpoint.stem + "_test_metrics.json")
    result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"metrics written to: {result_path}")


if __name__ == "__main__":
    main()
