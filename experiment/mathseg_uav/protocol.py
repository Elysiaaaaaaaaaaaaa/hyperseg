"""Shared data, losses, metrics and reproducibility contracts."""

import hashlib
import json
import random
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Sampler

from hyperseg_uav.data import UAVDataset


CLASS_NAMES = ("Ignore", "Background", "Building", "Road", "Water", "Barren",
               "Vegetation", "Agricultural", "Vehicle")


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    temporary.replace(path)


def resolve_dirs(root, image_dir=None, mask_dir=None):
    if (image_dir is None) != (mask_dir is None):
        raise ValueError("Provide both --image-dir and --mask-dir")
    if image_dir is not None:
        candidates = [(Path(image_dir), Path(mask_dir))]
    else:
        root = Path(root)
        candidates = [(root / p / "images", root / p / "masks")
                      for p in ("train", "train/train", "low_altitude_2026/train")]
    for images, masks in candidates:
        if images.is_dir() and masks.is_dir():
            return images, masks
    raise FileNotFoundError("No labelled data found; pass --image-dir and --mask-dir")


def read_splits(directory, images, masks):
    splits = {}
    for name in ("train", "val", "test"):
        entries = [s.strip() for s in (Path(directory) / f"{name}.txt").read_text(encoding="utf-8").splitlines() if s.strip()]
        if any(Path(s).name != s or s in (".", "..") for s in entries):
            raise ValueError(f"Only image stems or PNG filenames are allowed in {name}")
        ids = [s[:-4] if s.endswith(".png") else s for s in entries]
        if not ids or len(ids) != len(set(ids)):
            raise ValueError(f"Empty or duplicate split: {name}")
        for stem in ids:
            for directory_path in (images, masks):
                if not (directory_path / f"{stem}.png").is_file():
                    raise FileNotFoundError(directory_path / f"{stem}.png")
        splits[name] = ids
    if any(set(splits[a]) & set(splits[b]) for a, b in (("train", "val"), ("train", "test"), ("val", "test"))):
        raise ValueError("train/val/test splits overlap after filename normalization")
    hashes = {name: hashlib.sha256("\n".join(ids).encode()).hexdigest() for name, ids in splits.items()}
    return splits, hashes


def read_mask(path):
    with Image.open(path) as image:
        if image.mode != "L":
            raise ValueError(f"Masks must be single-channel L PNGs (got {image.mode}): {path}")
        array = np.asarray(image)
        if array.ndim != 2 or not np.issubdtype(array.dtype, np.integer):
            raise ValueError(f"Expected a single-channel integer mask: {path}")
        if array.min() < 0 or array.max() > 8:
            raise ValueError(f"Mask labels must be in 0..8: {path}")
        return array.copy()


def scan_masks(images, masks, ids):
    counts = np.zeros(9, dtype=np.int64)
    for stem in ids:
        mask = read_mask(masks / f"{stem}.png")
        with Image.open(images / f"{stem}.png") as image:
            if image.size != (mask.shape[1], mask.shape[0]):
                raise ValueError(f"Image/mask size mismatch: {stem}")
        counts += np.bincount(mask.ravel(), minlength=9)
    return counts.tolist()


class CheckedDataset(UAVDataset):
    def __init__(self, *args, seed=3407, **kwargs):
        super().__init__(*args, **kwargs)
        self.seed = seed
        self.checked = set()

    def __getitem__(self, index):
        epoch, index = index if isinstance(index, tuple) else (0, index)
        path = self.paths[index]
        if index not in self.checked:
            mask = read_mask(self.mask_dir / path.name)
            with Image.open(path) as image:
                if image.size != (mask.shape[1], mask.shape[0]):
                    raise ValueError(f"Image/mask size mismatch: {path}")
            self.checked.add(index)
        # Augmentation depends on epoch/index, not worker scheduling or prefetch.
        previous = random.getstate()
        random.seed(self.seed + epoch * len(self) + index)
        try:
            return super().__getitem__(index)
        finally:
            random.setstate(previous)


class UpdateBatchSampler(Sampler):
    """Resume directly at a microbatch without replaying image decoding."""

    def __init__(self, length, batch_size, seed, start_microbatch, total_microbatches):
        self.length, self.batch_size, self.seed = length, batch_size, seed
        self.start, self.total = start_microbatch, total_microbatches
        self.per_epoch = length // batch_size
        if self.per_epoch < 1:
            raise ValueError("Training split is smaller than the batch size")

    def __len__(self):
        return self.total - self.start

    def __iter__(self):
        old_epoch, order = -1, None
        for microbatch in range(self.start, self.total):
            epoch, offset = divmod(microbatch, self.per_epoch)
            if epoch != old_epoch:
                order = torch.randperm(self.length, generator=torch.Generator().manual_seed(self.seed + epoch)).tolist()
                old_epoch = epoch
            begin = offset * self.batch_size
            yield [(epoch, index) for index in order[begin:begin + self.batch_size]]


def make_loader(images, masks, ids, workers=0, size=None, seed=3407, batch_size=1,
                start_microbatch=0, total_microbatches=None):
    training = total_microbatches is not None
    dataset = CheckedDataset(images, masks, ids, size=size, training=training, seed=seed)
    common = dict(num_workers=workers, pin_memory=torch.cuda.is_available(),
                  generator=torch.Generator().manual_seed(seed), persistent_workers=workers > 0)
    if training:
        sampler = UpdateBatchSampler(len(dataset), batch_size, seed, start_microbatch, total_microbatches)
        return DataLoader(dataset, batch_sampler=sampler, **common)
    return DataLoader(dataset, batch_size=1, shuffle=False, **common)


def boundary_map(labels, valid):
    edge = torch.zeros_like(valid)
    horizontal = (labels[:, :, 1:] != labels[:, :, :-1]) & valid[:, :, 1:] & valid[:, :, :-1]
    vertical = (labels[:, 1:] != labels[:, :-1]) & valid[:, 1:] & valid[:, :-1]
    edge[:, :, 1:] |= horizontal
    edge[:, :, :-1] |= horizontal
    edge[:, 1:] |= vertical
    edge[:, :-1] |= vertical
    # Exclude the 3x3 neighborhood of Ignore; outside-image padding is not Ignore.
    safe = valid & (F.max_pool2d((~valid).float()[:, None], 3, 1, 1)[:, 0] == 0)
    return edge & safe, safe


def class_weights(counts, progress, gamma_max=0.5):
    counts = torch.as_tensor(counts, dtype=torch.float32)
    weights = torch.ones_like(counts)
    present = counts[1:] > 0
    frequencies = counts[1:][present] / counts[1:].sum().clamp_min(1)
    values = frequencies.clamp_min(1e-8).pow(-gamma_max * min(1.0, max(0.0, progress)))
    values = values / values.mean().clamp_min(1e-8)
    weights[1:][present] = values.clamp(0.25, 4.0)
    weights[0] = 0
    return weights


def segmentation_loss(output, target, weights=None):
    """Same CE + Dice + 0.1 boundary objective for M0--M4; M4 weights CE only."""
    logits = output["logits"].float()
    boundary = output["boundary"].float().squeeze(1)
    valid = target != 0
    if not valid.any():
        return (logits.sum() + boundary.sum()) * 0
    ce = F.cross_entropy(logits, target, ignore_index=0, weight=weights, reduction="none")[valid]
    denominator = valid.sum() if weights is None else weights[target[valid]].sum()
    ce = ce.sum() / denominator.clamp_min(1e-8)
    probabilities = logits.softmax(1)
    dice = []
    for cls in range(1, 9):
        truth = target[valid] == cls
        if truth.any():
            prediction = probabilities[:, cls][valid]
            dice.append(1 - (2 * (prediction * truth).sum() + 1) / (prediction.sum() + truth.sum() + 1))
    edge, safe = boundary_map(target, valid)
    bce = (F.binary_cross_entropy_with_logits(boundary[safe], edge[safe].float())
           if safe.any() else boundary.sum() * 0)
    return ce + torch.stack(dice).mean() + 0.1 * bce


def confusion_metrics(confusion):
    matrix = confusion.double()
    intersection = matrix.diag()
    union = matrix.sum(0) + matrix.sum(1) - intersection
    present = union[1:] > 0
    return {
        "mIoU": float((intersection[1:][present] / union[1:][present]).mean()) if present.any() else None,
        "per_class_iou": {str(c): float(intersection[c] / union[c]) if union[c] > 0 else None for c in range(1, 9)},
        "confusion": confusion.cpu().tolist(), "class_names": CLASS_NAMES,
    }


@torch.inference_mode()
def evaluate(model, loader, device):
    model.eval()
    confusion = torch.zeros(9, 9, dtype=torch.int64, device=device)
    # Micro boundary F1 at exact native pixels; no tolerance/dilation matching.
    boundary_counts = torch.zeros(3, dtype=torch.int64, device=device)
    for batch in loader:
        logits = model(batch["image"].to(device))["logits"]
        if not torch.isfinite(logits).all():
            raise FloatingPointError("Non-finite evaluation logits")
        prediction = logits.argmax(1)
        target = batch["mask"].to(device)
        valid = target != 0
        encoded = target[valid] * 9 + prediction[valid]
        confusion += torch.bincount(encoded, minlength=81).reshape(9, 9)
        actual, _ = boundary_map(target, valid)
        predicted, _ = boundary_map(prediction, valid)
        boundary_counts += torch.stack(((actual & predicted).sum(), predicted.sum(), actual.sum()))
    metrics = confusion_metrics(confusion)
    tp, predicted, actual = boundary_counts.tolist()
    metrics["boundary_f1_exact"] = 2 * tp / (predicted + actual) if predicted + actual else None
    metrics["samples"] = len(loader.dataset)
    return metrics


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False


def rng_state():
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if torch.cuda.is_available() and state["cuda"]:
        torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])
