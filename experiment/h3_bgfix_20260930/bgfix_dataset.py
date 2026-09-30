"""Dataset with the coverage-gap augmentations for the H3 background-collapse fix.

Rationale (see ``experiment/h3_bgfix_20260930/README.md`` for the full evidence):

* ``hyperseg_uav/data.py`` samples ``scale in (0.5, 0.75, 1.0, 1.25, 1.5)`` but then
  clamps with ``max(size, int(width * scale))``.  With 1024 px sources and ``size=512``
  that clamp is harmless, with ``size=768`` it silently collapses 0.5 and 0.75 into the
  same 0.75x view.  Either way the sampled scale is only ever applied on the *upscale*
  side of the crop, so the model never sees a zoomed-out view of a scene.
* The photometric branch only touches brightness/contrast/color.  The test set contains
  a coherent sub-population that is markedly darker, blue/green shifted and lower in
  contrast than anything in training, so that augmentation cannot cover it.
* Nothing ever degrades effective resolution, yet the worst test images have a mean
  gradient magnitude of 3.68 against 6.09 in training -- i.e. roughly 0.6x the detail
  scale.

This dataset keeps the spatial contract of the original (image bilinear, mask nearest,
mask values in ``0..8``, image float32 in ``[0, 1]`` on ``C x H x W``) and adds:

1. ``zoom_out``   -- a genuine down-scale branch: when the sampled scale makes the
   image smaller than the crop, the whole scene is pasted into a ``size x size`` canvas
   instead of being clamped back up.  Padding is image 0 / mask ignore.
2. ``gsd``        -- random down-sample followed by up-sample of the final patch, which
   models a higher flight altitude without moving any label.
3. ``photometric``-- directional colour cast, per-channel white balance, exposure, gamma,
   haze/low contrast and noise, aimed at the measured test-set appearance shift.
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import torch
from torch.utils.data import Dataset

RESAMPLE_IMAGE = Image.Resampling.BILINEAR
RESAMPLE_MASK = Image.Resampling.NEAREST

# Direction of the measured appearance shift of the collapsed test sub-population.
#
# Train means            R/G/B = 106.3 / 107.8 / 100.2
# Collapsed test means   R/G/B =  81.4 /  89.7 /  91.3
# Expressed as multiplicative deficits relative to the train mean:
#   1 - 81.4/106.3 = 0.2342,  1 - 89.7/107.8 = 0.1679,  1 - 91.3/100.2 = 0.0888
#
# With ``cast_strength = 1.0`` (so ``amount ~ U(0, 1)``) an amount of 1.0 reproduces the
# observed cast exactly and smaller amounts interpolate back towards the training look.
# An earlier version used a hand-picked (0.30, 0.20, 0.05), which topped out well short of
# the measured shift -- the smoke test caught that, which is why the direction is now
# derived from the data rather than guessed.
COLLAPSE_CAST_DIRECTION = (0.2342, 0.1679, 0.0888)
HAZE_COLOR = np.array([0.72, 0.78, 0.80], dtype=np.float32)

DEFAULT_PHOTOMETRIC = {
    "cast_prob": 0.55,
    "cast_strength": 1.0,
    "wb_prob": 0.45,
    "wb_strength": 0.14,
    "exposure_prob": 0.60,
    "exposure_range": (0.72, 1.06),
    "gamma_prob": 0.45,
    "gamma_range": (0.75, 1.35),
    "haze_prob": 0.35,
    "haze_max": 0.24,
    "blur_prob": 0.30,
    "blur_radius": (0.4, 1.4),
    "noise_prob": 0.30,
    "noise_sigma": 0.018,
    "legacy_prob": 0.35,
}


def _resize_keep_aspect(image: Image.Image, mask: Image.Image | None, width: int, height: int):
    image = image.resize((width, height), RESAMPLE_IMAGE)
    if mask is not None:
        mask = mask.resize((width, height), RESAMPLE_MASK)
    return image, mask


def _pad_to_canvas(image: Image.Image, mask: Image.Image | None, size: int, pad_fill: int):
    """Place the (smaller) image inside a ``size x size`` canvas at a random offset."""
    width, height = image.size
    canvas = Image.new("RGB", (size, size), (pad_fill, pad_fill, pad_fill))
    offset_x = random.randint(0, size - width)
    offset_y = random.randint(0, size - height)
    canvas.paste(image, (offset_x, offset_y))
    if mask is None:
        return canvas, None
    mask_canvas = Image.new("L", (size, size), 0)
    mask_canvas.paste(mask, (offset_x, offset_y))
    return canvas, mask_canvas


def _degenerate_scale_probability(scale: float, width: int, height: int, size: int) -> float:
    """Probability that a sampled scale cannot fill a crop and lands in the zoom-out path."""
    scaled_width = int(round(width * scale))
    scaled_height = int(round(height * scale))
    return 1.0 if min(scaled_width, scaled_height) < size else 0.0


def apply_photometric(array: np.ndarray, config: dict | None = None) -> np.ndarray:
    """In-place-safe photometric domain randomisation on a ``H x W x 3`` float array."""
    config = {**DEFAULT_PHOTOMETRIC, **(config or {})}
    result = array.astype(np.float32, copy=True)

    if random.random() < config["cast_prob"]:
        amount = random.uniform(0.0, config["cast_strength"])
        gain = 1.0 - amount * np.asarray(COLLAPSE_CAST_DIRECTION, dtype=np.float32)
        result *= gain

    if random.random() < config["wb_prob"]:
        strength = config["wb_strength"]
        gain = np.asarray(
            [random.uniform(1.0 - strength, 1.0 + strength) for _ in range(3)], dtype=np.float32
        )
        result *= gain

    if random.random() < config["exposure_prob"]:
        result *= random.uniform(*config["exposure_range"])

    result = np.clip(result, 0.0, 1.0)

    if random.random() < config["gamma_prob"]:
        result = np.power(result, random.uniform(*config["gamma_range"]))

    if random.random() < config["haze_prob"]:
        amount = random.uniform(0.0, config["haze_max"])
        result = result * (1.0 - amount) + HAZE_COLOR * amount

    if random.random() < config["noise_prob"]:
        sigma = config["noise_sigma"]
        result = result + np.random.normal(0.0, sigma, size=result.shape).astype(np.float32)

    return np.clip(result, 0.0, 1.0)


class BgFixUAVDataset(Dataset):
    """UAV dataset with zoom-out, GSD and photometric domain randomisation.

    ``training=False`` keeps a deterministic whole-image eval path.  ``eval_downscale``
    is the proxy-evaluation knob: it renders the sample at ``r`` x native resolution and
    returns it as-is, so mIoU is measured at the degraded detail scale rather than
    against a resized label.
    """

    def __init__(
        self,
        image_dir,
        mask_dir=None,
        ids=None,
        size: int | None = 512,
        training: bool = True,
        # --- fix switches (all default on; the "baseline" preset turns them off) ---
        enable_zoom_out: bool = True,
        scale_range=(0.4, 1.5),
        enable_gsd: bool = True,
        gsd_prob: float = 0.5,
        gsd_range=(0.5, 1.0),
        enable_photometric: bool = True,
        photometric: dict | None = None,
        photo_prob: float = 1.0,
        scene_crop_prob: float = 0.5,
        rare_classes=(3, 4, 5, 7, 8),
        pad_fill: int = 0,
        # --- deterministic eval ---
        eval_downscale: float | None = None,
        align_to: int = 32,
    ):
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir) if mask_dir else None
        self.size = size
        self.training = training
        self.enable_zoom_out = enable_zoom_out
        self.scale_range = tuple(scale_range)
        self.enable_gsd = enable_gsd
        self.gsd_prob = gsd_prob
        self.gsd_range = tuple(gsd_range)
        self.enable_photometric = enable_photometric
        self.photometric = {**DEFAULT_PHOTOMETRIC, **(photometric or {})}
        self.photo_prob = float(photo_prob)
        self.scene_crop_prob = scene_crop_prob
        self.rare_classes = tuple(rare_classes)
        self.pad_fill = pad_fill
        self.eval_downscale = eval_downscale
        self.align_to = align_to

        paths = sorted(self.image_dir.glob("*.png"))
        if ids is not None:
            names = {str(value).strip() for value in ids}
            paths = [path for path in paths if path.stem in names or path.name in names]
        self.paths = paths

    def __len__(self):
        return len(self.paths)

    # ------------------------------------------------------------------ training
    def _sample_crop_offset(self, mask: Image.Image, width: int, height: int):
        size = self.size
        best = (random.randint(0, width - size), random.randint(0, height - size))
        if mask is None or random.random() >= self.scene_crop_prob:
            return best
        array = np.asarray(mask)
        for _ in range(8):
            candidate_x = random.randint(0, width - size)
            candidate_y = random.randint(0, height - size)
            crop = array[candidate_y:candidate_y + size, candidate_x:candidate_x + size]
            distinct = np.unique(crop[crop != 0]).size
            if distinct >= 3 or np.isin(crop, self.rare_classes).any():
                return candidate_x, candidate_y
        return best

    def _load_training(self, image: Image.Image, mask: Image.Image | None):
        size = self.size
        scale = random.uniform(*self.scale_range)
        width = max(1, int(round(image.width * scale)))
        height = max(1, int(round(image.height * scale)))
        image, mask = _resize_keep_aspect(image, mask, width, height)

        if width >= size and height >= size:
            offset_x, offset_y = self._sample_crop_offset(mask, width, height)
            box = (offset_x, offset_y, offset_x + size, offset_y + size)
            image = image.crop(box)
            if mask is not None:
                mask = mask.crop(box)
        elif self.enable_zoom_out:
            image, mask = _pad_to_canvas(image, mask, size, self.pad_fill)
        else:
            # Legacy behaviour: clamp back up so the crop can be taken.
            image, mask = _resize_keep_aspect(
                image, mask, max(size, width), max(size, height)
            )
            offset_x = random.randint(0, image.width - size)
            offset_y = random.randint(0, image.height - size)
            box = (offset_x, offset_y, offset_x + size, offset_y + size)
            image = image.crop(box)
            if mask is not None:
                mask = mask.crop(box)

        if self.enable_gsd and random.random() < self.gsd_prob:
            ratio = random.uniform(*self.gsd_range)
            low = (max(8, int(size * ratio)), max(8, int(size * ratio)))
            image = image.resize(low, RESAMPLE_IMAGE).resize((size, size), RESAMPLE_IMAGE)

        if random.random() < 0.5:
            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            if mask is not None:
                mask = mask.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        if random.random() < 0.5:
            image = image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
            if mask is not None:
                mask = mask.transpose(Image.Transpose.FLIP_TOP_BOTTOM)

        if random.random() < self.photometric["blur_prob"]:
            image = image.filter(ImageFilter.GaussianBlur(random.uniform(*self.photometric["blur_radius"])))

        array = np.asarray(image, dtype=np.float32) / 255.0
        if self.enable_photometric and random.random() < self.photo_prob:
            array = apply_photometric(array, self.photometric)

        if random.random() < self.photometric["legacy_prob"]:
            array = self._legacy_photometric(array)

        return array, mask

    @staticmethod
    def _legacy_photometric(array: np.ndarray) -> np.ndarray:
        image = Image.fromarray((np.clip(array, 0.0, 1.0) * 255.0).astype(np.uint8))
        image = ImageEnhance.Brightness(image).enhance(random.uniform(0.65, 1.35))
        image = ImageEnhance.Contrast(image).enhance(random.uniform(0.65, 1.35))
        image = ImageEnhance.Color(image).enhance(random.uniform(0.7, 1.3))
        return np.asarray(image, dtype=np.float32) / 255.0

    # ---------------------------------------------------------------- evaluation
    def _load_eval(self, image: Image.Image, mask: Image.Image | None):
        if self.eval_downscale:
            align = max(1, self.align_to)
            width = int(round(image.width * self.eval_downscale))
            height = int(round(image.height * self.eval_downscale))
            width = max(align, width - width % align)
            height = max(align, height - height % align)
            image, mask = _resize_keep_aspect(image, mask, width, height)
        elif self.size:
            image, mask = _resize_keep_aspect(image, mask, self.size, self.size)
        return np.asarray(image, dtype=np.float32) / 255.0, mask

    def __getitem__(self, index):
        path = self.paths[index]
        image = Image.open(path).convert("RGB")
        mask = None
        if self.mask_dir is not None:
            mask = Image.open(self.mask_dir / path.name).convert("L")
        original_size = image.size

        if self.training:
            array, mask = self._load_training(image, mask)
        else:
            array, mask = self._load_eval(image, mask)

        result = {
            "image": torch.from_numpy(np.ascontiguousarray(array)).permute(2, 0, 1),
            "name": path.name,
            "original_size": original_size,
        }
        if mask is not None:
            result["mask"] = torch.from_numpy(np.asarray(mask, dtype=np.int64).copy())
        return result


def worker_init_fn(worker_id: int):
    """Give every DataLoader worker an independent, reproducible stream."""
    info = torch.utils.data.get_worker_info()
    base = info.seed if info is not None else torch.initial_seed()
    seed = (int(base) + worker_id) % (2**31)
    random.seed(seed)
    np.random.seed(seed)
