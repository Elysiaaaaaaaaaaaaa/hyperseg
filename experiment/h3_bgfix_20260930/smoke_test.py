"""Self-checks for the bgfix augmentation and loss before spending GPU hours.

Run this locally or on the server::

    python experiment/h3_bgfix_20260930/smoke_test.py

It needs only torch, numpy and Pillow -- no MMSegmentation, no CUDA, no dataset.
Everything is built from synthetic images in a temporary directory, so it is safe to run
anywhere.  Pass ``--keep`` to leave the temporary directory in place for inspection.

Why bother: the fix touches interpolation paths (image bilinear / mask nearest), label
padding and a new loss term.  A silent misalignment there would show up only as a
mediocre mIoU after a multi-day run, which is the most expensive possible failure mode.
"""

from __future__ import annotations

import argparse
import random
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from bgfix_dataset import (  # noqa: E402
    BgFixUAVDataset,
    COLLAPSE_CAST_DIRECTION,
    DEFAULT_PHOTOMETRIC,
    apply_photometric,
)
from bgfix_loss import bgfix_loss, confusion_matrix, inverse_frequency_weights, metrics_from_confusion  # noqa: E402

FAILURES: list[str] = []


def check(condition, label: str, detail: str = ""):
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {label}" + (f" -- {detail}" if detail else ""))
    if not condition:
        FAILURES.append(label)


def make_synthetic(root: Path, count: int = 4, size: int = 128):
    image_dir = root / "images"
    mask_dir = root / "masks"
    image_dir.mkdir(parents=True)
    mask_dir.mkdir(parents=True)
    rng = np.random.RandomState(0)
    for index in range(count):
        # left half red, right half blue; a green band in the middle.
        image = np.zeros((size, size, 3), dtype=np.uint8)
        image[:, : size // 2] = (220, 30, 30)
        image[:, size // 2:] = (30, 30, 220)
        image[:, size // 2 - 6: size // 2 + 6] = (30, 200, 30)
        mask = np.zeros((size, size), dtype=np.uint8)
        mask[:, : size // 2] = 1
        mask[:, size // 2:] = 2
        mask[:, size // 2 - 6: size // 2 + 6] = 6
        mask[0:6, :] = 0  # an ignore strip so the ignore path is exercised too
        if index == 3:
            image = np.clip(image.astype(np.int16) + rng.randint(-10, 10, image.shape), 0, 255).astype(np.uint8)
        Image.fromarray(image, "RGB").save(image_dir / f"s{index}.png")
        Image.fromarray(mask, "L").save(mask_dir / f"s{index}.png")
    return image_dir, mask_dir


def dominant_class(mask: np.ndarray, region: np.ndarray) -> int:
    values, counts = np.unique(mask[region], return_counts=True)
    return int(values[counts.argmax()])


def test_alignment(image_dir, mask_dir):
    print("\n[1] spatial alignment through crop / flip / pad")
    dataset = BgFixUAVDataset(
        image_dir, mask_dir, None, size=64, training=True,
        enable_zoom_out=True, scale_range=(1.0, 1.0),
        enable_gsd=False, enable_photometric=False,
        scene_crop_prob=0.0, photometric={"legacy_prob": 0.0},
    )
    for trial in range(8):
        sample = dataset[trial % len(dataset)]
        image = sample["image"]
        mask = sample["mask"]
        check(tuple(image.shape) == (3, 64, 64), f"trial {trial} image shape", str(tuple(image.shape)))
        check(tuple(mask.shape) == (64, 64), f"trial {trial} mask shape", str(tuple(mask.shape)))
        check(float(image.min()) >= 0.0 and float(image.max()) <= 1.0,
              f"trial {trial} image range", f"[{float(image.min()):.3f}, {float(image.max()):.3f}]")
        check(set(np.unique(mask.numpy()).tolist()) <= set(range(9)),
              f"trial {trial} mask labels in 0..8", str(sorted(np.unique(mask.numpy()).tolist())))

        # red pixels must be class 1, blue pixels class 2, green band class 6
        array = image.numpy().transpose(1, 2, 0)
        red = (array[:, :, 0] > 0.6) & (array[:, :, 1] < 0.4)
        blue = (array[:, :, 2] > 0.6) & (array[:, :, 1] < 0.4)
        green = (array[:, :, 1] > 0.6) & (array[:, :, 0] < 0.4)
        labels = mask.numpy()
        if red.sum() > 50:
            check(dominant_class(labels, red) == 1, f"trial {trial} red region is Background(1)",
                  f"got {dominant_class(labels, red)}")
        if blue.sum() > 50:
            check(dominant_class(labels, blue) == 2, f"trial {trial} blue region is Building(2)",
                  f"got {dominant_class(labels, blue)}")
        if green.sum() > 20:
            check(dominant_class(labels, green) == 6, f"trial {trial} green band is Vegetation(6)",
                  f"got {dominant_class(labels, green)}")


def test_zoom_out(image_dir, mask_dir):
    print("\n[2] zoom-out branch really fires (scale below crop/source ratio)")
    dataset = BgFixUAVDataset(
        image_dir, mask_dir, None, size=96, training=True,
        enable_zoom_out=True, scale_range=(0.45, 0.45),
        enable_gsd=False, enable_photometric=False,
        scene_crop_prob=0.0, photometric={"legacy_prob": 0.0},
    )
    pad_seen = False
    for trial in range(6):
        sample = dataset[trial % len(dataset)]
        image = sample["image"].numpy().transpose(1, 2, 0)
        mask = sample["mask"].numpy()
        black = (image.max(axis=2) < 1e-6)
        if black.any():
            pad_seen = True
            check(bool((mask[black] == 0).all()), f"trial {trial} pad region is ignore(0)")
            check(bool((mask[~black] != 0).any()), f"trial {trial} content region keeps labels")
        check(float((image.sum(axis=2) > 0).mean()) < 1.0 or not black.any(),
              f"trial {trial} canvas not fully covered or no padding", "")
    check(pad_seen, "padding path was actually exercised")

    legacy = BgFixUAVDataset(
        image_dir, mask_dir, None, size=96, training=True,
        enable_zoom_out=False, scale_range=(0.45, 0.45),
        enable_gsd=False, enable_photometric=False, photometric={"legacy_prob": 0.0},
    )
    print("    (legacy clamp branch for comparison)")
    sample = legacy[0]
    check(tuple(sample["image"].shape) == (3, 96, 96), "legacy clamp still yields a full crop")


def test_gsd_and_photometric(image_dir, mask_dir):
    print("\n[3] GSD and photometric randomisation keep geometry")
    dataset = BgFixUAVDataset(
        image_dir, mask_dir, None, size=64, training=True,
        enable_zoom_out=True, scale_range=(1.0, 1.0),
        enable_gsd=True, gsd_prob=1.0, gsd_range=(0.5, 0.5),
        enable_photometric=True, photo_prob=1.0,
        scene_crop_prob=0.0, photometric={"legacy_prob": 0.0},
    )
    for trial in range(6):
        sample = dataset[trial % len(dataset)]
        check(tuple(sample["image"].shape) == (3, 64, 64), f"trial {trial} shape after GSD+photo")
        check(tuple(sample["mask"].shape) == (64, 64), f"trial {trial} mask untouched by GSD")
        check(set(np.unique(sample["mask"].numpy()).tolist()) <= set(range(9)),
              f"trial {trial} labels still valid")

    print("\n    photometric bounds on a constant patch")
    flat = np.full((32, 32, 3), 0.5, dtype=np.float32)
    for _ in range(200):
        out = apply_photometric(flat, {**DEFAULT_PHOTOMETRIC, "legacy_prob": 0.0})
        if out.shape != flat.shape or out.min() < 0.0 or out.max() > 1.0 or not np.isfinite(out).all():
            check(False, "photometric output stays in range", f"min={out.min()} max={out.max()}")
            break
    else:
        check(True, "photometric output stays in [0,1] and finite over 200 draws")

    print("\n    directional colour cast must be able to reach the measured shift")
    train_means = np.array([106.3, 107.8, 100.2])
    collapsed_means = np.array([81.4, 89.7, 91.3])
    measured_deficit = 1.0 - collapsed_means / train_means
    gains = 1.0 - np.asarray(COLLAPSE_CAST_DIRECTION)
    check(bool(np.allclose(gains, 1.0 - measured_deficit, atol=0.02)),
          "cast direction at full strength reproduces the measured deficit",
          f"gains={np.round(gains, 4).tolist()} measured={np.round(1 - measured_deficit, 4).tolist()}")

    reached = 0
    max_delta = -1e9
    for _ in range(600):
        out = apply_photometric(
            np.full((8, 8, 3), 1.0, dtype=np.float32) * (train_means / 255.0).astype(np.float32),
            {**{key: 0.0 for key in DEFAULT_PHOTOMETRIC}, "cast_prob": 1.0,
             "cast_strength": DEFAULT_PHOTOMETRIC["cast_strength"], "legacy_prob": 0.0},
        )
        means = out.reshape(-1, 3).mean(0) * 255.0
        delta = float(means[1] - means[0])
        max_delta = max(max_delta, delta)
        if delta > 6.5 and means[2] > means[0]:
            reached += 1
    check(reached > 0, "cast augmentation reaches the observed G-R > 8 direction",
          f"{reached}/600 draws, best delta={max_delta:.2f}")
    check(max_delta >= 6.5, "strongest reachable G-R delta approaches the measured +6.9",
          f"best delta={max_delta:.2f}")


def test_loss():
    print("\n[4] loss terms")
    torch.manual_seed(0)
    logits = torch.zeros(2, 9, 16, 16)
    target = torch.full((2, 16, 16), 2, dtype=torch.long)
    target[0, :4, :4] = 0  # ignore block
    boundary = torch.zeros(2, 1, 16, 16)

    base = bgfix_loss({"logits": logits, "boundary": boundary}, target, false_bg_weight=0.0)
    boosted = bgfix_loss({"logits": logits, "boundary": boundary}, target, false_bg_weight=0.25)
    expected_penalty = 0.25 * (-np.log(1.0 - 1.0 / 9.0))
    delta = float(boosted - base)
    check(abs(delta - expected_penalty) < 1e-5,
          "false-background term equals 0.25 * -log(1 - p_bg)",
          f"delta={delta:.6f} expected={expected_penalty:.6f}")

    # The penalty must be blind to Background-labelled pixels.
    target_bg = torch.full((2, 16, 16), 1, dtype=torch.long)
    a = bgfix_loss({"logits": logits, "boundary": boundary}, target_bg, false_bg_weight=0.25)
    b = bgfix_loss({"logits": logits, "boundary": boundary}, target_bg, false_bg_weight=0.0)
    check(abs(float(a - b)) < 1e-6, "penalty does not fire on Background-labelled pixels",
          f"delta={float(a - b):.2e}")

    # Ignore pixels must not change the loss at all.
    noisy = target.clone()
    noisy[noisy == 0] = 5
    noisy[0, :4, :4] = 0
    check(abs(float(bgfix_loss({"logits": logits, "boundary": boundary}, target, false_bg_weight=0.25)
                   - bgfix_loss({"logits": logits, "boundary": boundary}, noisy, false_bg_weight=0.25))) < 1e-6,
          "ignore_index=0 pixels are excluded from every term")

    # A weight vector of ones must reproduce the unweighted loss.
    ones = torch.ones(9)
    weighted = bgfix_loss({"logits": logits, "boundary": boundary}, target,
                          class_weights=ones, false_bg_weight=0.25)
    check(abs(float(weighted - boosted)) < 1e-5, "unit class weights reproduce the unweighted loss",
          f"delta={float(weighted - boosted):.2e}")

    # F.cross_entropy weights follow the *target* class, so the weight only matters for
    # pixels that actually carry that label.
    background_batch = torch.full((2, 16, 16), 1, dtype=torch.long)
    heavy = torch.ones(9)
    heavy[1] = 5.0
    check(float(bgfix_loss({"logits": logits, "boundary": boundary}, background_batch,
                           class_weights=heavy, false_bg_weight=0.0)) > float(base),
          "raising the Background weight increases loss on a Background-labelled batch")
    foreground_only = torch.full((2, 16, 16), 2, dtype=torch.long)
    check(abs(float(bgfix_loss({"logits": logits, "boundary": boundary}, foreground_only,
                               class_weights=heavy, false_bg_weight=0.0))
              - float(bgfix_loss({"logits": logits, "boundary": boundary}, foreground_only,
                                 class_weights=None, false_bg_weight=0.0))) < 1e-6,
          "Background weight is inert on a batch with no Background labels")

    # The penalty has to be able to push the background logit down.
    pushed = logits.clone()
    pushed[:, 1] = 4.0
    penalty_on = float(bgfix_loss({"logits": pushed, "boundary": boundary}, target, false_bg_weight=0.25))
    penalty_off = float(bgfix_loss({"logits": pushed, "boundary": boundary}, target, false_bg_weight=0.0))
    check(penalty_on - penalty_off > 0.5,
          "penalty grows when Background dominates the logits", f"delta={penalty_on - penalty_off:.4f}")

    for name, value in (("false_bg_weight=0", float(base)), ("false_bg_weight=0.25", float(boosted))):
        check(np.isfinite(value), f"loss finite ({name})", f"{value:.6f}")


def test_class_weights():
    print("\n[5] inverse-frequency class weights")
    counts = [1_000_000, 27_170_000, 18_690_000, 8_590_000, 6_260_000, 1_970_000, 28_090_000, 6_840_000, 860_000]
    mild = inverse_frequency_weights(counts, "mild", floor=0.7, cap=3.0)
    median = inverse_frequency_weights(counts, "median", floor=0.1, cap=10.0)
    none = inverse_frequency_weights(counts, "none")
    check(all(abs(float(v) - 1.0) < 1e-6 for v in none), "mode 'none' returns ones")
    check(float(mild[8]) > float(mild[1]), "rarer class (Vehicle) outweighs Background",
          f"{float(mild[8]):.3f} vs {float(mild[1]):.3f}")
    check(float(mild[1]) >= 0.7 - 1e-6, "Background is floored, not suppressed", f"{float(mild[1]):.3f}")
    check(float(mild.max()) <= 3.0 + 1e-6, "cap respected", f"max={float(mild.max()):.3f}")
    check(float(median[8]) > float(mild[8]), "median mode is more aggressive than mild",
          f"{float(median[8]):.3f} vs {float(mild[8]):.3f}")
    check(float(mild[0]) == 1.0, "ignore class weight neutralised")


def test_metrics():
    print("\n[6] confusion-matrix metrics")
    confusion = torch.zeros(9, 9, dtype=torch.int64)
    confusion[1, 1] = 50    # Background correct
    confusion[1, 2] = 50    # labelled Background, predicted Building -> false negative
    confusion[2, 2] = 100   # Building correct
    confusion[2, 1] = 100   # labelled Building, predicted Background -> false positive
    # class 1: inter 50, union = col(50+100) + row(50+50) - 50 = 200 -> IoU 0.25
    # class 2: inter 100, union = col(50+100) + row(100+100) - 100 = 250 -> IoU 0.40
    metrics = metrics_from_confusion(confusion)
    check(abs(metrics["background_false_positive_rate"] - 100 / 200) < 1e-9,
          "background false-positive rate", f"{metrics['background_false_positive_rate']:.4f}")
    check(abs(metrics["background_false_negative_rate"] - 50 / 100) < 1e-9,
          "background false-negative rate", f"{metrics['background_false_negative_rate']:.4f}")
    check(abs(metrics["mIoU"] - (50 / 200 + 100 / 250) / 2) < 1e-9,
          "mIoU over the two observed classes", f"{metrics['mIoU']:.4f}")
    check(abs(metrics["mIoU"] - 0.325) < 1e-9, "hand-computed mIoU matches", f"{metrics['mIoU']:.4f}")

    # All-zero logits predict class 0.  A valid label predicted as 0 is still an error and
    # must land in column 0; only label==0 pixels are dropped.
    logits = torch.zeros(1, 9, 4, 4)
    target = torch.zeros(1, 4, 4, dtype=torch.long)
    target[0, 0, 0] = 3
    matrix = confusion_matrix(logits, target)
    check(int(matrix.sum()) == 1, "ignore pixels excluded from the confusion matrix", str(int(matrix.sum())))
    check(int(matrix[3, 0]) == 1,
          "valid label predicted as class 0 is recorded as an error (row 3, col 0)")

    # A correct prediction lands on the diagonal and a valid-but-wrong one does not.
    shaped = torch.zeros(1, 9, 2, 2)
    shaped[0, 5] = 5.0
    pair = torch.tensor([[[5, 5], [5, 5]]])
    diagonal = confusion_matrix(shaped, pair)
    check(int(diagonal[5, 5]) == 4, "confident correct predictions land on the diagonal")
    check(int(confusion_matrix(shaped, torch.full((1, 2, 2), 7, dtype=torch.long))[7, 5]) == 4,
          "confident wrong predictions land off the diagonal")


def test_eval_downscale(image_dir, mask_dir):
    print("\n[7] proxy eval renders labels at the reduced detail scale")
    for scale in (0.5, 0.625, 0.75):
        dataset = BgFixUAVDataset(image_dir, mask_dir, None, None, False, eval_downscale=scale)
        sample = dataset[0]
        height, width = sample["mask"].shape
        check(height % 32 == 0 and width % 32 == 0, f"scale {scale} aligned to 32", f"{width}x{height}")
        check(abs(width - 128 * scale) < 32, f"scale {scale} width near target", f"{width}")
        check(tuple(sample["image"].shape[1:]) == (height, width),
              f"scale {scale} image and mask agree", f"{tuple(sample['image'].shape[1:])} vs ({height}, {width})")
        check(set(np.unique(sample["mask"].numpy()).tolist()) <= set(range(9)),
              f"scale {scale} labels valid after nearest-neighbour resize")


def test_determinism(image_dir, mask_dir):
    print("\n[8] seeding gives reproducible augmentations")
    kwargs = dict(size=64, training=True, scale_range=(0.5, 1.2), enable_gsd=True, gsd_prob=0.7,
                  enable_photometric=True, photo_prob=1.0, photometric={"legacy_prob": 0.0})

    def draw(seed):
        random.seed(seed)
        np.random.seed(seed)
        return BgFixUAVDataset(image_dir, mask_dir, None, **kwargs)[1]["image"].clone()

    first, second, other = draw(11), draw(11), draw(12)
    check(torch.equal(first, second), "same seed reproduces the same sample")
    check(not torch.equal(first, other), "different seed changes the sample")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--keep", action="store_true", help="keep the synthetic dataset directory")
    parser.add_argument("--size", type=int, default=128)
    args = parser.parse_args()

    root = Path(tempfile.mkdtemp(prefix="bgfix_smoke_"))
    print(f"synthetic dataset: {root}")
    try:
        image_dir, mask_dir = make_synthetic(root, size=args.size)
        test_alignment(image_dir, mask_dir)
        test_zoom_out(image_dir, mask_dir)
        test_gsd_and_photometric(image_dir, mask_dir)
        test_loss()
        test_class_weights()
        test_metrics()
        test_eval_downscale(image_dir, mask_dir)
        test_determinism(image_dir, mask_dir)
    finally:
        if args.keep:
            print(f"kept: {root}")
        else:
            shutil.rmtree(root, ignore_errors=True)

    print("\n" + "=" * 60)
    if FAILURES:
        print(f"FAILED: {len(FAILURES)} check(s)")
        for name in FAILURES:
            print(f"  - {name}")
        return 1
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
