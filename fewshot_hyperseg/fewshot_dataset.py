"""
fewshot_dataset_v2.py
Fast episodic dataset for Mondstadt UAV segmentation.

Key fixes:
- Reads every mask once while building the index.
- Caches per-class pixel counts; _get_candidates never reopens masks.
- Uses precomputed class membership sets for fast query sampling.
- Creates a deterministic train/validation split instead of accidentally
  using the same full dataset for both.
- Supports query_size > 1.
- Keeps the same episode dictionary API.
"""
import os
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
from torchvision.transforms import ColorJitter

DEFAULT_DATA_ROOT = str(Path(__file__).resolve().parents[1] / "dataset")
NUM_CLASSES = 9
IGNORE_INDEX = 0
CLASS_NAMES = {
    0: "Ignore", 1: "Background", 2: "Building", 3: "Road",
    4: "Water", 5: "Barren", 6: "Vegetation",
    7: "Agricultural", 8: "Vehicle",
}
DEFAULT_CLASS_IDS = [2, 3, 4, 5, 6, 7, 8]


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id):
    seed = torch.initial_seed() % (2**32)
    random.seed(seed)
    np.random.seed(seed)


class FewShotTransform:
    def __init__(self, image_size=512, target_image_size=None, train=False, color_jitter=True):
        self.image_size = image_size
        self.target_image_size = target_image_size or image_size
        self.train = train
        self.jitter = ColorJitter(
            brightness=0.15, contrast=0.15,
            saturation=0.10, hue=0.02
        ) if train and color_jitter else None

    def __call__(self, image, mask):
        if self.train:
            if random.random() < 0.5:
                image = TF.hflip(image)
                mask = TF.hflip(mask)
            if random.random() < 0.5:
                image = TF.vflip(image)
                mask = TF.vflip(mask)
            if self.jitter is not None:
                image = self.jitter(image)

        image = TF.resize(
            image, [self.image_size, self.image_size],
            interpolation=TF.InterpolationMode.BILINEAR,
        )
        mask = TF.resize(
            mask, [self.target_image_size, self.target_image_size],
            interpolation=TF.InterpolationMode.NEAREST,
        )
        image = TF.to_tensor(image)
        mask = torch.from_numpy(np.asarray(mask, dtype=np.int64).copy())
        return image, mask


class MondstadtFileIndex:
    def __init__(self, data_root=DEFAULT_DATA_ROOT, seed=42,
                 val_ratio=0.1, verify_masks=True):
        self.data_root = data_root
        self.image_dir = os.path.join(data_root, "train", "train", "images")
        self.mask_dir = os.path.join(data_root, "train", "train", "masks")

        if not os.path.isdir(self.image_dir):
            raise FileNotFoundError(self.image_dir)
        if not os.path.isdir(self.mask_dir):
            raise FileNotFoundError(self.mask_dir)

        files = sorted(
            [f for f in os.listdir(self.image_dir)
             if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))],
            key=str.lower,
        )
        if not files:
            raise RuntimeError("No training images found")

        self.image_paths = []
        self.mask_paths = []
        self.class_pixel_counts = []
        self.class_to_images = {c: [] for c in range(NUM_CLASSES)}

        print("Building FAST Mondstadt index...")
        for filename in files:
            ip = os.path.join(self.image_dir, filename)
            mp = os.path.join(self.mask_dir, filename)
            if not os.path.exists(mp):
                if verify_masks:
                    raise FileNotFoundError(f"Missing mask: {mp}")
                continue

            mask = np.asarray(Image.open(mp).convert("L"), dtype=np.uint8)
            if mask.min() < 0 or mask.max() >= NUM_CLASSES:
                raise ValueError(f"Invalid labels in {mp}")

            idx = len(self.image_paths)
            self.image_paths.append(ip)
            self.mask_paths.append(mp)

            counts = np.bincount(mask.reshape(-1), minlength=NUM_CLASSES)
            counts = counts[:NUM_CLASSES].astype(np.int64)
            self.class_pixel_counts.append(counts)

            for c in range(1, NUM_CLASSES):
                if counts[c] > 0:
                    self.class_to_images[c].append(idx)

        self.class_pixel_counts = np.stack(self.class_pixel_counts, axis=0)
        self.num_images = len(self.image_paths)

        rng = random.Random(seed)
        indices = list(range(self.num_images))
        rng.shuffle(indices)

        val_n = max(1, int(round(self.num_images * val_ratio)))
        val_set = set(indices[:val_n])

        # Guarantee every requested foreground class exists in both splits.
        for c in DEFAULT_CLASS_IDS:
            train_c = [i for i in self.class_to_images[c] if i not in val_set]
            val_c = [i for i in self.class_to_images[c] if i in val_set]
            if not train_c:
                val_set.remove(val_c[0])
            elif not val_c:
                val_set.add(train_c[0])

        self.split_indices = {
            "train": [i for i in range(self.num_images) if i not in val_set],
            "val": [i for i in range(self.num_images) if i in val_set],
        }

        print(f"Images: {self.num_images}")
        print(f"Train : {len(self.split_indices['train'])}")
        print(f"Val   : {len(self.split_indices['val'])}")
        for c in range(1, NUM_CLASSES):
            print(f"  {c}: {CLASS_NAMES[c]:12s} "
                  f"{len(self.class_to_images[c]):5d}")

    def subset_class_to_images(self, split):
        allowed = set(self.split_indices[split])
        return {
            c: [i for i in self.class_to_images[c] if i in allowed]
            for c in range(NUM_CLASSES)
        }


class UAVFewShotDataset(Dataset):
    def __init__(
        self,
        data_root=DEFAULT_DATA_ROOT,
        n_way=7,
        k_shot=1,
        query_size=2,
        image_size=512,
        episodes_per_epoch=1000,
        class_ids=None,
        train=True,
        split=None,
        seed=42,
        min_pixels=100,
        val_ratio=0.1,
        target_image_size=None,
        verify_masks=True,
        cache_index=None,
        **kwargs,
    ):
        super().__init__()
        self.n_way = n_way
        self.k_shot = k_shot
        self.query_size = query_size
        self.image_size = image_size
        self.target_image_size = target_image_size or image_size
        self.episodes_per_epoch = episodes_per_epoch
        self.train = train if split is None else split == "train"
        self.split = "train" if self.train else "val"
        self.seed = seed
        self.min_pixels = min_pixels
        self.class_ids = (
            list(map(int, class_ids)) if class_ids is not None
            else DEFAULT_CLASS_IDS.copy()
        )

        if len(self.class_ids) != n_way:
            raise ValueError("n_way must equal len(class_ids)")

        # DataLoader workers each get their own dataset copy, but building
        # the 7000-image index is cheap compared with repeatedly opening
        # thousands of masks per episode.
        self.index = MondstadtFileIndex(
            data_root, seed=seed, val_ratio=val_ratio,
            verify_masks=verify_masks
        )
        self.indices = self.index.split_indices[self.split]
        self.allowed = set(self.indices)
        self.class_to_images = self.index.subset_class_to_images(self.split)

        self.valid_candidates = {}
        for c in self.class_ids:
            arr = self.class_to_images[c]
            if min_pixels > 0:
                arr = [
                    i for i in arr
                    if self.index.class_pixel_counts[i, c] >= min_pixels
                ]
            if len(arr) < k_shot + query_size:
                raise RuntimeError(
                    f"Class {c} has only {len(arr)} usable images in {self.split}; "
                    f"need {k_shot + query_size}"
                )
            self.valid_candidates[c] = tuple(arr)

        self.target_images = sorted({
            i for c in self.class_ids for i in self.class_to_images[c]
        })
        self.target_set = set(self.target_images)

        self.transform = FewShotTransform(
            image_size=image_size,
            target_image_size=self.target_image_size,
            train=self.train,
            color_jitter=self.train,
        )

        print(
            f"FewShotDataset[{self.split}] "
            f"N={n_way} K={k_shot} Q={query_size} "
            f"episodes={episodes_per_epoch}"
        )

    def __len__(self):
        return self.episodes_per_epoch

    def _sample_episode(self, episode_id):
        rng = random.Random(self.seed + episode_id)
        support = []
        support_set = set()

        for c in self.class_ids:
            candidates = list(self.valid_candidates[c])
            selected = []
            rng.shuffle(candidates)
            for idx in candidates:
                if idx not in support_set:
                    selected.append(idx)
                    support_set.add(idx)
                    if len(selected) == self.k_shot:
                        break
            if len(selected) < self.k_shot:
                raise RuntimeError(f"Cannot sample support for class {c}")
            support.extend(selected)

        # Fast query sampling: no repeated "idx in list" scans.
        preferred = [
            i for i in self.target_images if i not in support_set
        ]
        other = [
            i for i in self.indices
            if i not in support_set and i not in self.target_set
        ]
        rng.shuffle(preferred)
        rng.shuffle(other)

        query = (preferred + other)[:self.query_size]
        if len(query) < self.query_size:
            raise RuntimeError("Not enough query images")

        return support, query

    def _load_pair(self, idx):
        image = Image.open(self.index.image_paths[idx]).convert("RGB")
        mask = Image.open(self.index.mask_paths[idx]).convert("L")
        return self.transform(image, mask)

    def __getitem__(self, episode_id):
        support_idx, query_idx = self._sample_episode(episode_id)

        support_images, support_masks, support_paths = [], [], []
        for idx in support_idx:
            im, ma = self._load_pair(idx)
            support_images.append(im)
            support_masks.append(ma)
            support_paths.append(self.index.image_paths[idx])

        query_images, query_masks, query_paths = [], [], []
        for idx in query_idx:
            im, ma = self._load_pair(idx)
            query_images.append(im)
            query_masks.append(ma)
            query_paths.append(self.index.image_paths[idx])

        return {
            "support_images": torch.stack(support_images),
            "support_masks": torch.stack(support_masks),
            "query_images": torch.stack(query_images),
            "query_masks": torch.stack(query_masks),
            "class_ids": torch.tensor(self.class_ids, dtype=torch.long),
            "episode_id": episode_id,
            "support_paths": support_paths,
            "query_paths": query_paths,
        }


# Compatibility alias
FewShotUAVDataset = UAVFewShotDataset
FewShotDataset = UAVFewShotDataset
UAVEpisodeDataset = UAVFewShotDataset
FewShotEpisodeDataset = UAVFewShotDataset
