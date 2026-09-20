"""Create color masks and source-image overlays for semantic predictions."""

import argparse
from pathlib import Path

from PIL import Image


PALETTE = [
    (0, 0, 0),        # ignore
    (80, 80, 80),      # background
    (220, 50, 47),    # building
    (38, 139, 210),   # road
    (42, 161, 152),   # water
    (181, 101, 29),   # barren
    (67, 160, 71),    # vegetation
    (238, 190, 39),   # agricultural
    (155, 89, 182),   # vehicle
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--alpha", type=float, default=0.45)
    args = parser.parse_args()
    if not args.predictions.is_dir():
        raise FileNotFoundError(f"Prediction directory does not exist: {args.predictions.resolve()}")
    if not args.images.is_dir():
        raise FileNotFoundError(f"Image directory does not exist: {args.images.resolve()}")
    mask_dir = args.output / "masks"
    overlay_dir = args.output / "overlays"
    mask_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    overlay_count = 0
    for pred_path in sorted(args.predictions.glob("*.png")):
        prediction = Image.open(pred_path).convert("L")
        colors = Image.new("RGB", prediction.size)
        colors.putdata([PALETTE[min(value, len(PALETTE) - 1)] for value in prediction.getdata()])
        colors.save(mask_dir / pred_path.name)
        image_path = args.images / pred_path.name
        if image_path.is_file():
            image = Image.open(image_path).convert("RGB")
            if image.size != prediction.size:
                image = image.resize(prediction.size, Image.Resampling.BILINEAR)
            overlay = Image.blend(image, colors, args.alpha)
            overlay.save(overlay_dir / pred_path.name)
            overlay_count += 1
        count += 1
    print(f"visualized {count} predictions")
    print(f"matched source images and wrote {overlay_count} overlays")
    if count and not overlay_count:
        raise RuntimeError(
            "No source images matched prediction filenames. "
            "Run from the project directory and pass both paths relative to it, "
            "or use absolute paths."
        )
    print(f"masks: {mask_dir}")
    print(f"overlays: {overlay_dir}")


if __name__ == "__main__":
    main()
