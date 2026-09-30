"""Unlabelled regression check on the competition test set (``test_2``).

    python experiment/h3_bgfix_20260930/regression_test2.py \
        --checkpoint runs/mask2former_uav/h3_bgfix_full_160k/best.pt \
        --output experiment/h3_bgfix_20260930/results/regression_full.json

There is no ground truth for ``test_2``, so this reports no accuracy figure.  It reports
the four quantities the original diagnosis was built on, so the fix can be judged on the
same axes:

1. predicted class shares against the training prior,
2. the per-image Background share distribution (median vs the collapsed tail),
3. the per-bucket decay table -- if the fix works, the classes that were being eaten
   should keep their share as Background rises, instead of collapsing to ~0,
4. the correlation between image detail level and Background share.

Pass ``--reference <zip|dir>`` to diff against an earlier prediction set (for example the
existing H3 or MiT-B3 submissions) so the change is visible as a delta, not a side-by-side
eyeball.

Mechanics: sliding-window logits with the same protocol as
``experiment/mask2former_uav/infer_h3_hyperseg.py`` (512 px window, 0.5 overlap) but the
predictions are kept in memory instead of written as 1300 PNGs.
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
MASK2FORMER = ROOT / "experiment" / "mask2former_uav"
for extra in (str(ROOT), str(HERE), str(MASK2FORMER)):
    if extra not in sys.path:
        sys.path.insert(0, extra)

from hyperseg_uav.swin_l_model import SwinLHyperSeg  # noqa: E402
from infer_h3_hyperseg import sliding_logits  # noqa: E402

CLASS_NAMES = {0: "Ignore", 1: "Background", 2: "Building", 3: "Road", 4: "Water",
               5: "Barren", 6: "Vegetation", 7: "Agricultural", 8: "Vehicle"}

# Measured on runs/splits/train.txt, see configs/train_class_pixel_counts.json.
TRAIN_PRIOR = {1: 27.17, 2: 18.69, 3: 8.59, 4: 6.26, 5: 1.97, 6: 28.09, 7: 6.84, 8: 0.86}
BG_BUCKETS = ((0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.01))


def load_model(checkpoint: Path, device):
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    config = dict(state.get("model_config", {}))
    config["pretrained_checkpoint"] = None
    model = SwinLHyperSeg(**config).to(device)
    model.load_state_dict(state["model"])
    model.eval()
    return model, state


def detail_level(path: Path) -> float:
    """Mean absolute gradient -- the proxy for ground sample distance used in diagnosis."""
    grey = np.asarray(Image.open(path).convert("L"), dtype=np.float32)
    return float(0.5 * (np.abs(np.diff(grey, axis=1)).mean() + np.abs(np.diff(grey, axis=0)).mean()))


def load_reference(path: Path, names):
    """Read a previous prediction set (zip or directory) as a stack of label arrays."""
    masks = []
    if path.suffix == ".zip":
        with zipfile.ZipFile(path) as archive:
            for name in names:
                masks.append(np.asarray(Image.open(io.BytesIO(archive.read(name)))))
    else:
        for name in names:
            masks.append(np.asarray(Image.open(path / name)))
    return np.stack(masks)


def describe(masks: np.ndarray):
    """Class shares, the per-image Background distribution and the bucket decay table."""
    total = masks.size
    counts = Counter()
    unique, frequencies = np.unique(masks, return_counts=True)
    for label, frequency in zip(unique.tolist(), frequencies.tolist()):
        counts[label] += frequency
    shares = {CLASS_NAMES.get(key, str(key)): 100.0 * counts[key] / total
              for key in sorted(counts) if key in CLASS_NAMES and key != 0}

    background = (masks == 1).mean(axis=(1, 2))
    distribution = {
        "mean": float(background.mean()),
        "median": float(np.median(background)),
        "p90": float(np.percentile(background, 90)),
        "max": float(background.max()),
        "count_gt_0.6": int((background > 0.6).sum()),
        "count_gt_0.8": int((background > 0.8).sum()),
        "count_eq_1.0": int((background >= 0.999).sum()),
        "images": int(len(background)),
    }

    buckets = []
    for low, high in BG_BUCKETS:
        mask = (background >= low) & (background < high)
        if not mask.any():
            continue
        subset = masks[mask]
        row = {"range": f"{low}-{high}", "images": int(mask.sum())}
        for cls in (1, 2, 3, 4, 5, 6, 7, 8):
            row[CLASS_NAMES[cls]] = float(100.0 * (subset == cls).mean())
        buckets.append(row)

    shares_row = {CLASS_NAMES[cls]: float(100.0 * (masks == cls).mean()) for cls in range(1, 9)}
    return {"class_shares": shares, "background_distribution": distribution,
            "buckets": buckets, "share_by_class": shares_row}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--input", type=Path,
                        default=ROOT / "dataset/low_altitude_2026/test_2/images")
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--overlap", type=float, default=0.5)
    parser.add_argument("--reference", type=Path, action="append", default=[],
                        help="previous prediction zip or directory to diff against")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    paths = sorted(args.input.glob("*.png"))
    if not paths:
        raise FileNotFoundError(f"no PNGs in {args.input}")
    names = [path.name for path in paths]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, state = load_model(args.checkpoint, device)
    print(f"checkpoint step={state.get('step')} recorded_best={state.get('best_mIoU')}")
    print(f"{len(paths)} images from {args.input}, window {args.size} overlap {args.overlap}")

    predictions = np.empty((len(paths), 0), dtype=np.uint8)
    stack = []
    for index, path in enumerate(paths, start=1):
        array = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
        image = torch.from_numpy(array).permute(2, 0, 1)[None].to(device)
        labels = sliding_logits(model, image, args.size, args.overlap).argmax(1)[0]
        stack.append(labels.cpu().numpy().astype(np.uint8))
        if index % 50 == 0 or index == len(paths):
            print(f"  {index}/{len(paths)}", flush=True)
    predictions = np.stack(stack)

    report = {
        "checkpoint": str(args.checkpoint),
        "step": state.get("step"),
        "input": str(args.input),
        "images": len(paths),
        "protocol": {"size": args.size, "overlap": args.overlap, "tta": False},
        "train_prior": {CLASS_NAMES[cls]: value for cls, value in TRAIN_PRIOR.items()},
        "prediction": describe(predictions),
    }

    details = np.array([detail_level(path) for path in paths])
    background = (predictions == 1).mean(axis=(1, 2))
    report["detail_vs_background"] = {
        "corr_detail_background": float(np.corrcoef(details, background)[0, 1]),
        "detail_mean": float(details.mean()),
        "detail_p05": float(np.percentile(details, 5)),
        "detail_p50": float(np.percentile(details, 50)),
    }

    print("\npredicted class shares vs training prior")
    print(f"  {'class':<13}{'predicted':>10}{'prior':>9}{'delta':>9}")
    for cls in range(1, 9):
        name = CLASS_NAMES[cls]
        predicted = report["prediction"]["share_by_class"][name]
        print(f"  {name:<13}{predicted:>9.2f}%{TRAIN_PRIOR[cls]:>8.2f}%{predicted - TRAIN_PRIOR[cls]:>+8.2f}")

    print("\nper-image Background share distribution")
    for key, value in report["prediction"]["background_distribution"].items():
        print(f"  {key:<16}{value if isinstance(value, int) else f'{value:.4f}'}")

    print("\nclass share by Background bucket")
    header = "  " + f"{'bucket':<12}{'n':>6}" + "".join(f"{CLASS_NAMES[cls][:4]:>8}" for cls in (1, 2, 3, 4, 5, 6, 7, 8))
    print(header)
    for row in report["prediction"]["buckets"]:
        line = "  " + f"{row['range']:<12}{row['images']:>6}" + "".join(
            f"{row[CLASS_NAMES[cls]]:>8.2f}" for cls in (1, 2, 3, 4, 5, 6, 7, 8))
        print(line)
    print(f"\ncorr(detail level, Background share) = {report['detail_vs_background']['corr_detail_background']:.3f}")

    for reference in args.reference:
        reference_masks = load_reference(reference, names)
        described = describe(reference_masks)
        report.setdefault("references", {})[str(reference)] = described
        print(f"\ndelta vs {reference.name}")
        print(f"  {'class':<13}{'new':>9}{'reference':>11}{'delta':>9}")
        for cls in range(1, 9):
            name = CLASS_NAMES[cls]
            new_value = report["prediction"]["share_by_class"][name]
            old_value = described["share_by_class"][name]
            print(f"  {name:<13}{new_value:>8.2f}%{old_value:>10.2f}%{new_value - old_value:>+8.2f}")
        old_distribution = described["background_distribution"]
        new_distribution = report["prediction"]["background_distribution"]
        print(f"  background >0.6: {new_distribution['count_gt_0.6']} vs {old_distribution['count_gt_0.6']}"
              f"   >0.8: {new_distribution['count_gt_0.8']} vs {old_distribution['count_gt_0.8']}")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"\nwritten: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
