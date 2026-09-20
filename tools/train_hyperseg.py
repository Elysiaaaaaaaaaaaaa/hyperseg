"""Train HyperSeg-UAV on the Mondstadt dataset layout."""

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hyperseg_uav import HyperSegUAV, UAVDataset, hyperseg_loss, mean_iou

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "dataset")
    parser.add_argument("--split-dir", type=Path, default=ROOT / "runs/splits")
    parser.add_argument("--out", type=Path, default=ROOT / "runs/hyperseg_best.pt")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--size", type=int, default=768)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--encoder-lr-multiplier", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def read_ids(split_dir, name):
    path = split_dir / f"{name}.txt"
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    image_dir = args.data / "train" / "train" / "images"
    mask_dir = args.data / "train" / "train" / "masks"
    train_set = UAVDataset(image_dir, mask_dir, read_ids(args.split_dir, "train"), args.size, True)
    val_set = UAVDataset(image_dir, mask_dir, read_ids(args.split_dir, "val"), args.size, False)
    loader_args = {"batch_size": args.batch_size, "num_workers": args.num_workers, "pin_memory": torch.cuda.is_available()}
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, **loader_args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HyperSegUAV(pretrained=args.pretrained).to(device)
    encoder_ids = {id(parameter) for parameter in model.encoder.parameters()}
    encoder_params = [parameter for parameter in model.parameters() if id(parameter) in encoder_ids]
    head_params = [parameter for parameter in model.parameters() if id(parameter) not in encoder_ids]
    optimizer = torch.optim.AdamW(
        [{"params": encoder_params, "lr": args.lr * args.encoder_lr_multiplier}, {"params": head_params, "lr": args.lr}],
        weight_decay=0.01,
    )
    amp_enabled = args.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    best_score = -1.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            image = batch["image"].to(device, non_blocking=True)
            target = batch["mask"].to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=amp_enabled):
                loss = hyperseg_loss(model(image), target)
            if not torch.isfinite(loss):
                continue
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            total_loss += float(loss.detach())
        model.eval()
        score_sum = 0.0
        batch_count = 0
        with torch.no_grad():
            for batch in val_loader:
                image = batch["image"].to(device, non_blocking=True)
                target = batch["mask"].to(device, non_blocking=True)
                score_sum += mean_iou(model(image)["logits"], target)
                batch_count += 1
        score = score_sum / max(1, batch_count)
        train_loss = total_loss / max(1, len(train_loader))
        print(f"epoch {epoch}/{args.epochs}: train_loss={train_loss:.4f}, val_mIoU={score:.4f}", flush=True)
        if score > best_score:
            best_score = score
            torch.save({"model": model.state_dict(), "model_config": model.model_config, "epoch": epoch, "val_miou": score}, args.out)

if __name__=='__main__': main()
