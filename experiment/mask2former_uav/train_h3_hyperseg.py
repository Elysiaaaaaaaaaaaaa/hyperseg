"""Train/evaluate H3: Swin-L with the repository's HyperSeg decoder."""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hyperseg_uav import UAVDataset, hyperseg_loss
from hyperseg_uav.swin_l_model import SwinLHyperSeg


DEFAULT_BACKBONE = (
    "https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/swin/"
    "swin_large_patch4_window12_384_22k_20220412-6580f57d.pth"
)


def resolve_dirs(data_root: Path, image_dir: Path | None, mask_dir: Path | None):
    if image_dir and mask_dir:
        return image_dir, mask_dir
    candidates = [
        (data_root / "train" / "images", data_root / "train" / "masks"),
        (data_root / "train" / "train" / "images", data_root / "train" / "train" / "masks"),
        (data_root / "low_altitude_2026" / "train" / "images",
         data_root / "low_altitude_2026" / "train" / "masks"),
    ]
    for candidate_image, candidate_mask in candidates:
        if candidate_image.is_dir() and candidate_mask.is_dir():
            return candidate_image, candidate_mask
    raise FileNotFoundError(
        "Could not find labelled data. Pass --image-dir and --mask-dir explicitly."
    )


def ids(split_dir: Path, name: str):
    path = split_dir / f"{name}.txt"
    values = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not values:
        raise ValueError(f"Empty split: {path}")
    return values


def make_loader(image_dir, mask_dir, sample_ids, size, training, batch_size, workers):
    dataset = UAVDataset(image_dir, mask_dir, sample_ids, size, training)
    if not len(dataset):
        raise ValueError(f"No images matched split ({len(sample_ids)} IDs) in {image_dir}")
    if len(dataset) != len(sample_ids) or len(set(sample_ids)) != len(sample_ids):
        raise ValueError("Split contains duplicates or missing images")
    for path in dataset.paths:
        if not (mask_dir / path.name).is_file():
            raise FileNotFoundError(mask_dir / path.name)
    if training and len(dataset) < batch_size:
        raise ValueError("Training dataset is smaller than batch size")
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=training,
        drop_last=training,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=workers > 0,
    )


@torch.no_grad()
def evaluate(model, loader, device, classes=9, ignore_index=0):
    confusion = torch.zeros(classes, classes, dtype=torch.int64, device=device)
    model.eval()
    for batch in loader:
        output = model(batch["image"].to(device, non_blocking=True))["logits"]
        target = batch["mask"].to(device, non_blocking=True)
        prediction = output.argmax(1)
        valid = (target != ignore_index) & (target >= 0) & (target < classes)
        encoded = target[valid] * classes + prediction[valid]
        confusion += torch.bincount(encoded, minlength=classes * classes).reshape(classes, classes)
    intersection = confusion.diag().float()
    union = confusion.sum(0).float() + confusion.sum(1).float() - intersection
    valid = union[1:] > 0
    per_class = {
        str(cls): float(intersection[cls] / union[cls]) if union[cls] else 0.0
        for cls in range(1, classes)
    }
    miou = float((intersection[1:][valid] / union[1:][valid]).mean()) if valid.any() else 0.0
    return {"mIoU": miou, "per_class_iou": per_class, "confusion": confusion.cpu().tolist()}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def set_poly_lr(optimizer, base_lr, backbone_multiplier, step, max_iters, power):
    factor = max(0.0, 1.0 - step / max_iters) ** power
    optimizer.param_groups[0]["lr"] = base_lr * backbone_multiplier * factor
    optimizer.param_groups[1]["lr"] = base_lr * factor


def save_checkpoint(path, model, optimizer, step, best, args, metrics):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    torch.save({
        "model": model.state_dict(),
        "model_config": model.model_config,
        "optimizer": optimizer.state_dict(),
        "step": step,
        "best_mIoU": best,
        "metrics": metrics,
        "experiment": "H3_swin_l_hyperseg",
        "args": vars(args),
    }, temporary)
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=ROOT / "dataset")
    parser.add_argument("--image-dir", type=Path)
    parser.add_argument("--mask-dir", type=Path)
    parser.add_argument("--split-dir", type=Path, default=ROOT / "runs/splits")
    parser.add_argument("--work-dir", type=Path, default=ROOT / "runs/mask2former_uav/h3_swin_l_hyperseg_512_seed3407")
    parser.add_argument("--backbone-checkpoint", default=DEFAULT_BACKBONE)
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--accumulative-counts", type=int, default=1)
    parser.add_argument("--max-iters", type=int, default=160000)
    parser.add_argument("--val-interval", type=int, default=2800)
    parser.add_argument("--val-size", type=int, default=0, help="0 keeps native validation resolution")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--backbone-lr-multiplier", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--with-checkpointing", action="store_true")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--eval-checkpoint", type=Path,
                        help="evaluate a saved H3 checkpoint and exit")
    parser.add_argument("--eval-split", choices=("val", "test"), default="test")
    args = parser.parse_args()
    if (args.accumulative_counts < 1 or args.max_iters < 1 or
            args.val_interval < 1 or args.batch_size < 1 or args.size < 1 or
            args.num_workers < 0 or args.val_size < 0):
        parser.error("iteration, batch, size, and worker arguments are invalid")
    set_seed(args.seed)
    image_dir, mask_dir = resolve_dirs(args.data_root, args.image_dir, args.mask_dir)
    split_sets = [set(ids(args.split_dir, name)) for name in ("train", "val", "test")]
    if any(split_sets[i] & split_sets[j] for i in range(3) for j in range(i)):
        raise ValueError("train/val/test splits overlap")
    train_loader = make_loader(image_dir, mask_dir, ids(args.split_dir, "train"), args.size, True, args.batch_size, args.num_workers)
    val_loader = make_loader(image_dir, mask_dir, ids(args.split_dir, "val"), args.val_size or None, False, 1, args.num_workers)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = None if args.no_pretrained or args.resume or args.eval_checkpoint else args.backbone_checkpoint
    model = SwinLHyperSeg(pretrained_checkpoint=checkpoint, with_cp=args.with_checkpointing).to(device)
    backbone_params = list(model.backbone.parameters())
    backbone_ids = {id(parameter) for parameter in backbone_params}
    head_params = [parameter for parameter in model.parameters() if id(parameter) not in backbone_ids]
    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": args.lr * args.backbone_lr_multiplier},
        {"params": head_params, "lr": args.lr},
    ], weight_decay=0.01, betas=(0.9, 0.999))
    start_step, best = 0, -1.0
    if args.resume:
        state = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        start_step, best = int(state.get("step", 0)), float(state.get("best_mIoU", -1.0))
    if args.eval_checkpoint:
        state = torch.load(args.eval_checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        eval_loader = make_loader(
            image_dir, mask_dir, ids(args.split_dir, args.eval_split),
            args.val_size or None, False, 1, args.num_workers)
        metrics = evaluate(model, eval_loader, device)
        metrics["checkpoint"] = str(args.eval_checkpoint)
        metrics["split"] = args.eval_split
        print(json.dumps(metrics, indent=2, ensure_ascii=False))
        output = args.eval_checkpoint.with_name(args.eval_checkpoint.stem + f"_{args.eval_split}_metrics.json")
        output.write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"metrics written to: {output}")
        return
    args.work_dir.mkdir(parents=True, exist_ok=True)
    configuration = {**vars(args), "image_dir": image_dir, "mask_dir": mask_dir,
                     "model_config": model.model_config,
                     "train_samples": len(train_loader.dataset), "val_samples": len(val_loader.dataset)}
    (args.work_dir / "config.json").write_text(
        json.dumps(configuration, default=str, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(configuration, default=str), flush=True)
    started = time.monotonic()
    train_iterator = iter(train_loader)
    for step in range(start_step, args.max_iters):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss_value = 0.0
        for _ in range(args.accumulative_counts):
            try:
                batch = next(train_iterator)
            except StopIteration:
                train_iterator = iter(train_loader)
                batch = next(train_iterator)
            loss = hyperseg_loss(
                model(batch["image"].to(device, non_blocking=True)),
                batch["mask"].to(device, non_blocking=True),
            ) / args.accumulative_counts
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at update {step + 1}")
            loss.backward()
            loss_value += float(loss.detach())
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        set_poly_lr(optimizer, args.lr, args.backbone_lr_multiplier, step + 1, args.max_iters, 0.9)
        update = step + 1
        if update % 50 == 0 or update == 1:
            print(f"iter {update}/{args.max_iters}: loss={loss_value:.5f} "
                  f"grad_norm={float(grad_norm):.4f} "
                  f"seconds_per_update={(time.monotonic()-started)/(update-start_step):.3f} "
                  f"peak_gpu_gb={torch.cuda.max_memory_allocated()/1e9 if device.type == 'cuda' else 0:.2f}", flush=True)
        if update % args.val_interval == 0 or update == args.max_iters:
            metrics = evaluate(model, val_loader, device)
            print(json.dumps({"iter": update, **metrics}, ensure_ascii=False), flush=True)
            (args.work_dir / "val_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
            if metrics["mIoU"] > best:
                best = metrics["mIoU"]
                save_checkpoint(args.work_dir / "best.pt", model, optimizer, update, best, args, metrics)
            save_checkpoint(args.work_dir / "last.pt", model, optimizer, update, best, args, metrics)
    print(f"completed: {args.work_dir}", flush=True)


if __name__ == "__main__":
    main()
