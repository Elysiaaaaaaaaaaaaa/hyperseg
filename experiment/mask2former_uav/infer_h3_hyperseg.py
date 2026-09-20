"""Export H3 Swin-L + HyperSeg predictions for an image directory."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hyperseg_uav.swin_l_model import SwinLHyperSeg


@torch.no_grad()
def sliding_logits(model, image, size, overlap):
    _, _, height, width = image.shape
    stride = max(1, int(size * (1.0 - overlap)))
    classes = model.model_config["classes"]
    output = image.new_zeros((1, classes, height, width))
    weights = image.new_zeros((1, 1, height, width))
    tops = sorted(set(range(0, max(1, height - size + 1), stride)) | {max(0, height - size)})
    lefts = sorted(set(range(0, max(1, width - size + 1), stride)) | {max(0, width - size)})
    for top in tops:
        for left in lefts:
            bottom, right = min(height, top + size), min(width, left + size)
            crop = image[:, :, top:bottom, left:right]
            crop_shape = crop.shape[-2:]
            if crop_shape != (size, size):
                crop = F.interpolate(crop, (size, size), mode="bilinear", align_corners=False)
            logits = model(crop)["logits"]
            logits = F.interpolate(logits, crop_shape, mode="bilinear", align_corners=False)
            output[:, :, top:bottom, left:right] += logits
            weights[:, :, top:bottom, left:right] += 1
    return output / weights.clamp_min(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--overlap", type=float, default=0.5)
    args = parser.parse_args()
    if not args.checkpoint.is_file() or not args.input.is_dir():
        raise FileNotFoundError("checkpoint and input directory must exist")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = dict(state.get("model_config", {}))
    config["pretrained_checkpoint"] = None
    model = SwinLHyperSeg(**config).to(device)
    model.load_state_dict(state["model"])
    model.eval()
    args.output.mkdir(parents=True, exist_ok=True)
    for path in sorted(args.input.glob("*.png")):
        array = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
        image = torch.from_numpy(array).permute(2, 0, 1)[None].to(device)
        prediction = sliding_logits(model, image, args.size, args.overlap).argmax(1)[0]
        Image.fromarray(prediction.cpu().numpy().astype(np.uint8), mode="L").save(args.output / path.name)


if __name__ == "__main__":
    main()
