"""Evaluate an H3 checkpoint on the labelled split at native and reduced detail scales.

    python experiment/h3_bgfix_20260930/eval_proxy.py \
        --checkpoint runs/mask2former_uav/h3_bgfix_full_160k/best.pt \
        --split val

Reports, for one or more checkpoints side by side:

* native whole-image mIoU (the protocol the published H3 checkpoint was selected under),
* proxy mIoU at each ``--proxy-scales`` value: the same labelled images rendered at a
  reduced detail scale, which is the one axis of the ``test_2`` shift that labelled data
  can reproduce,
* the Background false-positive and false-negative rates, which is what actually moved on
  ``test_2`` (Background ate every other class, so false positives dominate).

The proxy is a necessary but not sufficient stand-in: it does not reproduce the colour
cast, the haze, or the new scene types.  Treat a proxy gain as evidence that the fix
works on the *detail-scale* margin, not as a prediction of the competition score.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
for extra in (str(ROOT), str(HERE)):
    if extra not in sys.path:
        sys.path.insert(0, extra)

from hyperseg_uav import UAVDataset  # noqa: E402
from hyperseg_uav.swin_l_model import SwinLHyperSeg  # noqa: E402
from bgfix_dataset import BgFixUAVDataset  # noqa: E402
from bgfix_loss import confusion_matrix, metrics_from_confusion  # noqa: E402

CLASS_NAMES = {1: "Background", 2: "Building", 3: "Road", 4: "Water",
               5: "Barren", 6: "Vegetation", 7: "Agricultural", 8: "Vehicle"}


def load_model(checkpoint: Path, device):
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    config = dict(state.get("model_config", {}))
    config["pretrained_checkpoint"] = None
    model = SwinLHyperSeg(**config).to(device)
    model.load_state_dict(state["model"])
    model.eval()
    return model, state


@torch.no_grad()
def evaluate_loader(model, loader, device, classes=9):
    confusion = torch.zeros(classes, classes, dtype=torch.int64, device=device)
    for batch in loader:
        logits = model(batch["image"].to(device, non_blocking=True))["logits"]
        confusion += confusion_matrix(logits, batch["mask"].to(device, non_blocking=True), classes=classes)
    return metrics_from_confusion(confusion)


def build_loader(image_dir, mask_dir, sample_ids, downscale, workers):
    if downscale is None:
        dataset = UAVDataset(image_dir, mask_dir, sample_ids, None, False)
    else:
        dataset = BgFixUAVDataset(image_dir, mask_dir, sample_ids, None, False, eval_downscale=downscale)
    return DataLoader(dataset, batch_size=1, shuffle=False, num_workers=workers,
                      pin_memory=torch.cuda.is_available())


def summarise(metrics, label):
    per_class = " ".join(
        f"{CLASS_NAMES[int(cls)]}={metrics['per_class_iou'][cls]:.3f}"
        for cls in sorted(metrics["per_class_iou"], key=lambda key: int(key))
    )
    return (f"  {label:<12} mIoU={metrics['mIoU']:.4f}  "
            f"bgFP={metrics['background_false_positive_rate']:.4f}  "
            f"bgFN={metrics['background_false_negative_rate']:.4f}\n      {per_class}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, action="append", required=True)
    parser.add_argument("--data-root", type=Path, default=ROOT / "dataset/low_altitude_2026")
    parser.add_argument("--split-dir", type=Path, default=ROOT / "runs/splits")
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--proxy-scales", default="0.5,0.625,0.75")
    parser.add_argument("--max-images", type=int, default=0, help="0 uses the whole split")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    image_dir = args.data_root / "train" / "images"
    mask_dir = args.data_root / "train" / "masks"
    if not image_dir.is_dir():
        raise FileNotFoundError(image_dir)
    names = [line.strip() for line in (args.split_dir / f"{args.split}.txt").read_text().splitlines() if line.strip()]
    if 0 < args.max_images < len(names):
        names = names[: args.max_images]
    scales = tuple(float(value) for value in args.proxy_scales.split(",") if value.strip())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaders = {"native": build_loader(image_dir, mask_dir, names, None, args.num_workers)}
    for scale in scales:
        loaders[f"proxy@{scale}"] = build_loader(image_dir, mask_dir, names, scale, args.num_workers)

    report = {"split": args.split, "images": len(names), "scales": list(scales), "checkpoints": {}}
    for checkpoint in args.checkpoint:
        model, state = load_model(checkpoint, device)
        print(f"\n{checkpoint}")
        print(f"  step={state.get('step')} recorded_best={state.get('best_mIoU')}")
        entry = {"step": state.get("step"), "recorded_best_mIoU": state.get("best_mIoU"), "metrics": {}}
        for label, loader in loaders.items():
            metrics = evaluate_loader(model, loader, device)
            entry["metrics"][label] = metrics
            print(summarise(metrics, label))
        proxy_values = [entry["metrics"][f"proxy@{scale}"]["mIoU"] for scale in scales]
        entry["proxy_mIoU_mean"] = float(np.mean(proxy_values)) if proxy_values else None
        entry["native_mIoU"] = entry["metrics"]["native"]["mIoU"]
        entry["proxy_gap"] = (
            float(entry["native_mIoU"] - entry["proxy_mIoU_mean"])
            if entry["proxy_mIoU_mean"] is not None else None
        )
        print(f"  proxy mean mIoU = {entry['proxy_mIoU_mean']:.4f}   "
              f"native - proxy gap = {entry['proxy_gap']:.4f}")
        report["checkpoints"][str(checkpoint)] = entry

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"\nwritten: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
