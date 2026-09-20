"""SUIM data discovery, RGB-mask decoding, and few-shot augmentation."""
from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance
import torch
from torch.utils.data import Dataset

from experiment.fewShot_SUIM.constants import CLASS_NAMES, NUM_CLASSES, SPLITS


@dataclass(frozen=True)
class Sample:
    split: str
    name: str
    image: Path
    mask: Path
    target_classes: tuple[int, ...] = ()

    @property
    def key(self) -> str:
        return f"{self.split}/{self.name}"


def _split_dir(root: Path, split: str) -> Path:
    canonical = SPLITS.get(split.lower())
    if canonical is None:
        raise ValueError(f"Unsupported SUIM split: {split}")
    path = Path(root) / canonical
    if not path.is_dir():
        raise FileNotFoundError(path)
    return path


def resolve_sample(root: Path, key: str, target_classes: tuple[int, ...] = ()) -> Sample:
    parts = key.split("/")
    if len(parts) != 2 or parts[0].lower() not in SPLITS or Path(parts[1]).name != parts[1]:
        raise ValueError(f"Invalid SUIM sample key: {key}")
    split = parts[0].lower()
    name = parts[1]
    if Path(name).suffix.lower() not in (".jpg", ".jpeg"):
        raise ValueError(f"SUIM image key must end in .jpg or .jpeg: {key}")
    folder = _split_dir(root, split)
    image = folder / "images" / name
    mask = folder / "masks" / f"{Path(name).stem}.bmp"
    if not image.is_file() or not mask.is_file():
        raise FileNotFoundError(f"Missing image/mask pair for {key}: {image}, {mask}")
    return Sample(SPLITS[split], name, image, mask, tuple(target_classes))


def discover_samples(root: Path, split: str) -> list[Sample]:
    split = split.lower()
    folder = _split_dir(root, split)
    image_dir, mask_dir = folder / "images", folder / "masks"
    if not image_dir.is_dir() or not mask_dir.is_dir():
        raise FileNotFoundError(f"Expected images/ and masks/ under {folder}")
    images = sorted(
        path for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in (".jpg", ".jpeg")
    )
    image_stems = {path.stem for path in images}
    mask_stems = {path.stem for path in mask_dir.glob("*.bmp") if path.is_file()}
    if mask_stems - image_stems:
        raise ValueError(f"Orphan top-level masks in {mask_dir}: {sorted(mask_stems - image_stems)[:5]}")
    samples = []
    for image in images:
        mask = mask_dir / f"{image.stem}.bmp"
        if not mask.is_file():
            raise FileNotFoundError(f"Missing mask for {image}: {mask}")
        samples.append(Sample(SPLITS[split], image.name, image, mask))
    if not samples:
        raise ValueError(f"No SUIM image/mask pairs found in {folder}")
    return samples


def decode_mask(mask_path: Path, image_size: tuple[int, int]) -> tuple[np.ndarray, dict[str, int | bool]]:
    """Decode SUIM RGB bits; repair the known masks whose bottom is 55 px too tall."""
    with Image.open(mask_path) as source:
        rgb = np.asarray(source.convert("RGB"), dtype=np.uint8)
    image_width, image_height = image_size
    mask_height, mask_width = rgb.shape[:2]
    repaired = False
    if (mask_width, mask_height) != image_size:
        if mask_width == image_width and mask_height == image_height + 55:
            rgb = rgb[:image_height, :image_width]
            repaired = True
        else:
            raise ValueError(
                f"Unsupported image/mask size mismatch for {mask_path}: "
                f"image={image_size}, mask={(mask_width, mask_height)}"
            )
    off_palette = np.any((rgb != 0) & (rgb != 255), axis=2)
    distance = np.minimum(rgb, 255 - rgb).max(axis=2)
    labels = (
        (rgb[..., 0] >= 128).astype(np.uint8) * 4
        + (rgb[..., 1] >= 128).astype(np.uint8) * 2
        + (rgb[..., 2] >= 128).astype(np.uint8)
    )
    audit = {
        "pixels": int(labels.size),
        "off_palette_pixels": int(off_palette.sum()),
        "far_from_palette_pixels": int((distance > 64).sum()),
        "size_repaired": repaired,
    }
    return labels, audit


def read_sample(sample: Sample) -> tuple[Image.Image, np.ndarray, dict[str, int | bool]]:
    with Image.open(sample.image) as source:
        image = source.convert("RGB")
    mask, audit = decode_mask(sample.mask, image.size)
    return image, mask, audit


class SUIMDataset(Dataset):
    def __init__(self, samples: list[Sample], crop_size: int | None, training: bool) -> None:
        self.samples = samples
        self.crop_size = crop_size
        self.training = training
        if training and (crop_size is None or crop_size < 1):
            raise ValueError("Training requires a positive crop size")

    def __len__(self) -> int:
        return len(self.samples)

    def _training_transform(self, image: Image.Image, mask_array: np.ndarray, sample: Sample):
        crop = int(self.crop_size)
        scale = max(random.uniform(0.75, 1.5), crop / min(image.size))
        size = max(crop, round(image.width * scale)), max(crop, round(image.height * scale))
        image = image.resize(size, Image.Resampling.BILINEAR)
        mask_image = Image.fromarray(mask_array, mode="L").resize(size, Image.Resampling.NEAREST)
        resized_mask = np.asarray(mask_image)

        target = random.choice(sample.target_classes) if sample.target_classes else None
        positions = np.argwhere(resized_mask == target) if target is not None else np.empty((0, 2), dtype=int)
        if len(positions):
            y, x = positions[random.randrange(len(positions))]
            left_min, left_max = max(0, int(x) - crop + 1), min(int(x), size[0] - crop)
            top_min, top_max = max(0, int(y) - crop + 1), min(int(y), size[1] - crop)
            left = random.randint(left_min, left_max)
            top = random.randint(top_min, top_max)
        else:
            left = random.randint(0, size[0] - crop)
            top = random.randint(0, size[1] - crop)
        box = left, top, left + crop, top + crop
        image, mask_image = image.crop(box), mask_image.crop(box)
        if random.random() < 0.5:
            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            mask_image = mask_image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        if random.random() < 0.7:
            image = ImageEnhance.Brightness(image).enhance(random.uniform(0.85, 1.15))
            image = ImageEnhance.Contrast(image).enhance(random.uniform(0.85, 1.15))
        if random.random() < 0.3:
            image = ImageEnhance.Color(image).enhance(random.uniform(0.9, 1.1))
        return image, np.asarray(mask_image, dtype=np.uint8)

    def __getitem__(self, index: int) -> dict[str, object]:
        sample = self.samples[index]
        image, mask, _ = read_sample(sample)
        if self.training:
            image, mask = self._training_transform(image, mask, sample)
        image_array = np.asarray(image, dtype=np.float32) / 255.0
        return {
            "image": torch.from_numpy(np.ascontiguousarray(image_array)).permute(2, 0, 1),
            "mask": torch.from_numpy(np.ascontiguousarray(mask, dtype=np.int64)),
            "name": sample.key,
        }


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)
