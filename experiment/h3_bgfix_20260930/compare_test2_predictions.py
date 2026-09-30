"""Compare several test_2 prediction sets on the axes the collapse diagnosis used.

    python experiment/h3_bgfix_20260930/compare_test2_predictions.py \
        --pred h3=experiment/mask2former_uav/test2_20260926/swin_l_hyperseg_复赛.zip \
        --pred baseline=.../pred_test2_baseline_20k \
        --pred loss=.../pred_test2_loss_20k \
        --pred aug=.../pred_test2_aug_20k \
        --pred full=.../pred_test2_full_20k \
        --images dataset/low_altitude_2026/test_2/images \
        --out-json experiment/h3_bgfix_20260930/results/test2_compare_20k.json \
        --montage experiment/h3_bgfix_20260930/results/test2_montage_20k.png

``test_2`` has no ground truth, so nothing here is an accuracy figure.  What it reports is
whether the collapse moved: the predicted class shares against the training prior, the
per-image Background share distribution, and -- for the samples where the reference model
had already collapsed -- whether the other classes got any pixels back.

Background is class 1 (27.8% of training pixels), not class 0; class 0 is the ignore index
and should not appear in a prediction at all.  A non-zero share for class 0 is reported as
``ignore_leak`` because it means the argmax is emitting the ignore label.

Accepts either a directory of same-named PNGs or a ``.zip`` / ``.tar.gz`` of one.  Only one
prediction set is held in memory at a time: five 1300x1024 sets would be 6.8 GB as uint8.
"""

from __future__ import annotations

import argparse
import io
import json
import tarfile
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

CLASSES = 9
BACKGROUND = 1
IGNORE = 0

# Class 0 is never drawn; the rest are distinguishable at a glance and stable across panels.
PALETTE = {
    1: (214, 214, 206),   # background / non-target hard surface
    2: (200, 60, 60),
    3: (230, 150, 60),
    4: (235, 205, 80),
    5: (110, 170, 90),
    6: (70, 130, 190),
    7: (150, 110, 190),
    8: (90, 200, 200),
}
CLASS_NAMES = {1: "bg", 2: "c2", 3: "c3", 4: "c4", 5: "c5", 6: "c6", 7: "c7", 8: "c8"}


def _open_member(path: Path):
    """Return (namelist_getter, reader) for a directory, zip or tar.gz of PNGs."""
    if path.is_dir():
        files = sorted(path.glob("*.png"))
        return lambda: [f.name for f in files], lambda name: (path / name).read_bytes()
    if path.suffix == ".zip":
        zf = zipfile.ZipFile(path)
        names = [n for n in zf.namelist() if n.lower().endswith(".png")]
        return lambda: [Path(n).name for n in names], lambda name: zf.read(
            next(n for n in names if Path(n).name == name))
    if path.name.endswith(".tar.gz") or path.suffix in (".tgz", ".tar"):
        tf = tarfile.open(path)
        names = [m.name for m in tf.getmembers() if m.name.lower().endswith(".png")]
        return lambda: [Path(n).name for n in names], lambda name: tf.extractfile(
            next(n for n in names if Path(n).name == name)).read()
    raise ValueError(f"unsupported prediction container: {path}")


def array_of(raw: bytes) -> np.ndarray:
    return np.asarray(Image.open(io.BytesIO(raw)).convert("L"), dtype=np.uint8)


def profile(path: Path) -> dict:
    """Per-image class counts plus a global histogram, streamed one image at a time."""
    names_fn, read_fn = _open_member(path)
    names = sorted(names_fn(), key=lambda n: (len(n), n))
    histogram = np.zeros(CLASSES, dtype=np.int64)
    per_image = []
    for name in names:
        array = array_of(read_fn(name))
        counts = np.bincount(array.ravel(), minlength=CLASSES)[:CLASSES]
        histogram += counts
        per_image.append((name, counts.tolist()))
    return {"path": str(path), "count": len(names), "histogram": histogram.tolist(),
            "per_image_counts": per_image}


def background_share(counts: list[int]) -> float:
    total = float(sum(counts))
    return counts[BACKGROUND] / total if total else 0.0


def bucket_table(sets, reference: str, edges=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0001)) -> dict:
    """Group images by the *reference* model's Background share, then look at every set inside.

    This is the part of the diagnosis that matters most: a fix that lowers the Background
    headline number by trading it for one other class is not a fix.  Holding the bucket
    membership fixed to the reference model's own collapse means each row is the same images
    seen by every model, so a class that used to vanish in the high-Background buckets either
    comes back or does not.
    """
    lookup = {name: counts for name, counts in sets[reference]["per_image_counts"]}
    ordered = sorted(lookup, key=lambda n: (len(n), n))
    rows = []
    for low, high in zip(edges, edges[1:]):
        members = [n for n in ordered if low <= background_share(lookup[n]) < high]
        row = {"range": [low, min(high, 1.0)], "images": len(members), "sets": {}}
        for name, record in sets.items():
            counts = {n: c for n, c in record["per_image_counts"]}
            present = [counts[n] for n in members if n in counts]
            if not present:
                continue
            stacked = np.array(present, dtype=np.float64)
            totals = stacked.sum(axis=1, keepdims=True)
            shares = np.divide(stacked, totals, out=np.zeros_like(stacked), where=totals > 0)
            row["sets"][name] = {
                "mean_background_share": float(shares[:, BACKGROUND].mean()),
                "mean_class_shares": {str(c): float(shares[:, c].mean()) for c in range(1, CLASSES)},
            }
        rows.append(row)
    return {"reference": reference, "rows": rows}


def shares(histogram: list[int]) -> dict[str, float]:
    total = float(sum(histogram))
    return {str(c): (histogram[c] / total if total else 0.0) for c in range(CLASSES)}


def distribution(values: np.ndarray) -> dict:
    return {
        "median": float(np.median(values)),
        "mean": float(values.mean()),
        "p90": float(np.percentile(values, 90)),
        "over_0.6": int((values > 0.6).sum()),
        "over_0.8": int((values > 0.8).sum()),
        "over_0.95": int((values > 0.95).sum()),
        "max": float(values.max()),
    }


def colorize(mask: np.ndarray) -> np.ndarray:
    """Render a label map as RGB.  Ignore stays black so a leak is impossible to miss."""
    rgb = np.zeros(mask.shape + (3,), dtype=np.uint8)
    for label, color in PALETTE.items():
        rgb[mask == label] = color
    return rgb


def overlay(image: np.ndarray, mask: np.ndarray, alpha: float = 0.55) -> np.ndarray:
    """Blend a colourised prediction over the source image."""
    base = image.astype(np.float32)
    tint = colorize(mask).astype(np.float32)
    painted = mask != IGNORE
    mixed = base * (1.0 - alpha * painted[..., None]) + tint * (alpha * painted[..., None])
    return mixed.clip(0, 255).astype(np.uint8)


def build_montage(predictions, images_dir: Path, samples, out_path: Path, cell: int = 240) -> None:
    """One row per sample: source image then each prediction overlaid on it."""
    columns = len(predictions) + 1
    canvas = np.full((len(samples) * cell, columns * cell, 3), 245, dtype=np.uint8)
    for row, name in enumerate(samples):
        source = np.asarray(Image.open(images_dir / name).convert("RGB").resize((cell, cell), Image.BILINEAR))
        canvas[row * cell:(row + 1) * cell, 0:cell] = source
        for column, (_, path) in enumerate(predictions, start=1):
            _, read_fn = _open_member(path)
            mask = array_of(read_fn(name))
            mask = np.asarray(Image.fromarray(mask, mode="L").resize((cell, cell), Image.NEAREST))
            canvas[row * cell:(row + 1) * cell,
                   column * cell:(column + 1) * cell] = overlay(source, mask)
        for column in range(1, columns):
            canvas[row * cell:(row + 1) * cell, column * cell - 1:column * cell + 1] = 120
    Image.fromarray(canvas).save(out_path)
    print(f"montage -> {out_path}  ({len(samples)} rows x {columns} columns)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pred", action="append", required=True, metavar="NAME=PATH")
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--prior", type=Path,
                        default=Path(__file__).resolve().parent / "configs/train_class_pixel_counts.json")
    parser.add_argument("--reference", help="name of the set whose collapsed images seed the montage")
    parser.add_argument("--reference-threshold", type=float, default=0.8)
    parser.add_argument("--montage", type=Path)
    parser.add_argument("--montage-samples", type=int, default=6)
    parser.add_argument("--out-json", type=Path)
    args = parser.parse_args()

    predictions = []
    for item in args.pred:
        name, _, raw = item.partition("=")
        if not raw:
            raise SystemExit(f"--pred needs NAME=PATH, got {item!r}")
        predictions.append((name, Path(raw)))

    prior = None
    if args.prior.is_file():
        blob = json.loads(args.prior.read_text(encoding="utf-8"))
        counts = blob["class_pixel_counts"]
        total = float(sum(counts.values()))
        prior = {str(c): counts[str(c)] / total for c in range(CLASSES)}

    results = {"prior": prior, "sets": {}}
    backgrounds = {}
    for name, path in predictions:
        record = profile(path)
        values = np.array([background_share(counts) for _, counts in record["per_image_counts"]])
        backgrounds[name] = record
        results["sets"][name] = {
            "path": record["path"],
            "images": record["count"],
            "class_shares": shares(record["histogram"]),
            "background_distribution": distribution(values),
        }
        print(f"{name:<12} images={record['count']:<5} "
              f"bg median={np.median(values):.3f} >0.8={int((values > 0.8).sum())} "
              f">0.6={int((values > 0.6).sum())}")

    if prior:
        print("\nclass shares (prediction vs training prior)")
        header = f"{'set':<12}" + "".join(f"{CLASS_NAMES[c]:>8}" for c in range(1, CLASSES))
        print(header)
        print(f"{'prior':<12}" + "".join(f"{prior[str(c)]:>8.4f}" for c in range(1, CLASSES)))
        for name, _ in predictions:
            row = results["sets"][name]["class_shares"]
            print(f"{name:<12}" + "".join(f"{row[str(c)]:>8.4f}" for c in range(1, CLASSES)))

    if len(predictions) > 1:
        base_name = predictions[0][0]
        print(f"\nper-image background delta vs {base_name} (negative = less collapsed)")
        base = {n: background_share(c) for n, c in backgrounds[base_name]["per_image_counts"]}
        for name, _ in predictions[1:]:
            current = {n: background_share(c) for n, c in backgrounds[name]["per_image_counts"]}
            common = sorted(set(base) & set(current))
            delta = np.array([current[k] - base[k] for k in common])
            recovered = sum(1 for k in common if base[k] > 0.8 and current[k] <= 0.8)
            worsened = sum(1 for k in common if base[k] <= 0.8 and current[k] > 0.8)
            print(f"  {name:<12} mean delta={delta.mean():+.4f}  "
                  f"left the >0.8 pool={recovered}  entered it={worsened}  (n={len(common)})")
            results["sets"][name]["delta_vs_reference"] = {
                "reference": base_name, "mean": float(delta.mean()),
                "left_collapsed_pool": recovered, "entered_collapsed_pool": worsened,
                "n": len(common)}

        table = bucket_table(backgrounds, base_name)
        results["buckets"] = table
        print(f"\nper-image buckets, membership fixed to {base_name}'s background share")
        print(f"{'bucket':<11}{'n':>5}  {'set':<12}" +
              "".join(f"{CLASS_NAMES[c]:>8}" for c in range(1, CLASSES)))
        for row in table["rows"]:
            if not row["images"]:
                continue
            label = f"{row['range'][0]:.1f}-{row['range'][1]:.1f}"
            for name, _ in predictions:
                entry = row["sets"].get(name)
                if not entry:
                    continue
                print(f"{label:<11}{row['images']:>5}  {name:<12}" +
                      "".join(f"{entry['mean_class_shares'][str(c)]:>8.4f}"
                              for c in range(1, CLASSES)))

    if args.montage:
        reference = args.reference or predictions[0][0]
        record = backgrounds[reference]
        ranked = sorted(((n, background_share(c)) for n, c in record["per_image_counts"]),
                        key=lambda item: item[1], reverse=True)
        samples = [name for name, value in ranked
                   if value > args.reference_threshold][:args.montage_samples]
        if not samples:
            samples = [name for name, _ in ranked[:args.montage_samples]]
        args.montage.parent.mkdir(parents=True, exist_ok=True)
        build_montage(predictions, args.images, samples, args.montage)
        results["montage_samples"] = samples

    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n",
                                 encoding="utf-8", newline="\n")
        print(f"\njson -> {args.out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
