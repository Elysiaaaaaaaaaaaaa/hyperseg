import random
from pathlib import Path
import numpy as np
from PIL import Image, ImageEnhance
import torch
from torch.utils.data import Dataset


class UAVDataset(Dataset):
    def __init__(
        self,
        image_dir,
        mask_dir=None,
        ids=None,
        size=1024,
        training=True,
        scene_crop_prob=0.3,
        rare_classes=(5, 7, 8),
    ):
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir) if mask_dir else None
        self.size, self.training = size, training
        self.scene_crop_prob, self.rare_classes = scene_crop_prob, tuple(rare_classes)
        paths = sorted(self.image_dir.glob("*.png"))
        if ids is not None:
            names = {str(x).strip() for x in ids}
            paths = [p for p in paths if p.stem in names or p.name in names]
        self.paths = paths

    def __len__(self): return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        image = Image.open(path).convert("RGB")
        mask = Image.open(self.mask_dir / path.name).convert("L") if self.mask_dir else None
        original_size = image.size
        if self.training:
            scale = random.choice((0.5, 0.75, 1.0, 1.25, 1.5))
            nw = max(self.size, int(image.width * scale))
            nh = max(self.size, int(image.height * scale))
            image = image.resize((nw, nh), Image.Resampling.BILINEAR)
            if mask:
                mask = mask.resize((nw, nh), Image.Resampling.NEAREST)
            x, y = random.randint(0, nw - self.size), random.randint(0, nh - self.size)
            if mask and random.random() < self.scene_crop_prob:
                arr = np.asarray(mask)
                for _ in range(8):
                    cx, cy = random.randint(0, nw - self.size), random.randint(0, nh - self.size)
                    crop = arr[cy:cy + self.size, cx:cx + self.size]
                    if np.unique(crop[crop != 0]).size >= 3 or np.isin(crop, self.rare_classes).any():
                        x, y = cx, cy
                        break
            image = image.crop((x, y, x + self.size, y + self.size))
            if mask:
                mask = mask.crop((x, y, x + self.size, y + self.size))
            if random.random() < 0.5:
                image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
                if mask:
                    mask = mask.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            if random.random() < 0.5:
                image = image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
                if mask:
                    mask = mask.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
            if random.random() < 0.7:
                image = ImageEnhance.Brightness(image).enhance(random.uniform(0.65, 1.35))
                image = ImageEnhance.Contrast(image).enhance(random.uniform(0.65, 1.35))
            if random.random() < 0.4:
                image = ImageEnhance.Color(image).enhance(random.uniform(0.7, 1.3))
        elif self.size:
            image = image.resize((self.size, self.size), Image.Resampling.BILINEAR)
            if mask: mask = mask.resize((self.size, self.size), Image.Resampling.NEAREST)
        array = np.asarray(image, dtype=np.float32) / 255.0
        result = {
            "image": torch.from_numpy(np.ascontiguousarray(array)).permute(2, 0, 1),
            "name": path.name,
            "original_size": original_size,
        }
        if mask:
            result["mask"] = torch.from_numpy(np.asarray(mask, dtype=np.int64).copy())
        return result
