"""Train M0--M4 or evaluate a checkpoint at native image resolution."""

import argparse
import json
import platform
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import transformers

from experiment.mathseg_uav.model import MathSegUAV, PRIOR_NAMES, VARIANTS
from experiment.mathseg_uav.protocol import (
    class_weights, evaluate, make_loader, read_splits, resolve_dirs, restore_rng,
    rng_state, scan_masks, seed_everything, segmentation_loss, write_json,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=VARIANTS)
    parser.add_argument("--data-root", type=Path, default=ROOT / "dataset")
    parser.add_argument("--image-dir", type=Path)
    parser.add_argument("--mask-dir", type=Path)
    parser.add_argument("--split-dir", type=Path, default=ROOT / "runs/splits")
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument("--model-name", default="nvidia/mit-b3")
    parser.add_argument("--allow-download", action="store_true", help="allow Hugging Face network loading")
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--prior-channels", nargs="+", choices=PRIOR_NAMES, default=list(PRIOR_NAMES))
    parser.add_argument("--prior-mode", choices=("normal", "zero", "shuffle"), default="normal")
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--accumulation", type=int, default=1)
    parser.add_argument("--max-updates", type=int, default=160000)
    parser.add_argument("--val-interval", type=int, default=2800)
    parser.add_argument("--val-size", type=int, default=0,
                        help="validation resize for smoke tests; 0 keeps native resolution")
    parser.add_argument("--save-interval", type=int, default=2800)
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument("--stop-after", type=int, help="stop/save at this absolute update; retain full LR horizon")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--backbone-lr-multiplier", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--gamma-max", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--device", default="cuda", help="cuda, cuda:0, or cpu (explicit smoke testing)")
    parser.add_argument("--amp", action="store_true", help="CUDA float16; FP32 is the default")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--resume", type=Path)
    group.add_argument("--eval-checkpoint", type=Path)
    group.add_argument("--check-only", action="store_true", help="scan labels/alignment in all three splits")
    parser.add_argument("--eval-split", choices=("val", "test"), default="test")
    args = parser.parse_args()
    positive = (args.width, args.size, args.batch_size, args.accumulation, args.max_updates,
                args.val_interval, args.save_interval, args.log_interval, args.lr, args.backbone_lr_multiplier)
    if min(positive) <= 0 or args.val_size < 0 or args.num_workers < 0 or args.gamma_max < 0 or args.weight_decay < 0:
        parser.error("Invalid nonpositive training parameter")
    if args.stop_after is not None and not 0 < args.stop_after <= args.max_updates:
        parser.error("--stop-after must be in 1..max-updates")
    if len(set(args.prior_channels)) != len(args.prior_channels):
        parser.error("--prior-channels contains duplicates")
    return args


def contract(args, hashes):
    # Paths and worker counts may change when moving a run to another machine.
    keys = ("variant", "width", "prior_channels", "prior_mode", "size", "batch_size", "accumulation",
            "max_updates", "val_size", "lr", "backbone_lr_multiplier", "weight_decay", "gamma_max", "seed", "amp")
    return {"version": 1, "splits": hashes, **{key: getattr(args, key) for key in keys}}


def save_checkpoint(path, model, optimizer, scaler, update, best, metrics, args, hashes, counts):
    temporary = path.with_suffix(".tmp")
    torch.save({
        "experiment": "mathseg_uav_v1", "model": model.state_dict(), "model_config": model.model_config,
        "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(), "update": update,
        "best_mIoU": best, "metrics": metrics, "rng": rng_state(),
        "contract": contract(args, hashes), "class_counts": counts,
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
    }, temporary)
    temporary.replace(path)


def load_state(path):
    # Only load trusted experiment files: optimizer/RNG state requires pickle.
    state = torch.load(path, map_location="cpu", weights_only=False)
    if state.get("experiment") != "mathseg_uav_v1":
        raise ValueError("Expected a MathSeg checkpoint; HyperSeg checkpoints are incompatible")
    return state


def main():
    args = parse_args()
    images, masks = resolve_dirs(args.data_root, args.image_dir, args.mask_dir)
    splits, hashes = read_splits(args.split_dir, images, masks)
    if args.check_only:
        counts = {name: scan_masks(images, masks, ids) for name, ids in splits.items()}
        print(json.dumps({"samples": {k: len(v) for k, v in splits.items()}, "split_hashes": hashes,
                          "class_counts": counts}, indent=2))
        return
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; use --device cpu only for small smoke tests")
    if args.amp and device.type != "cuda":
        raise ValueError("--amp requires CUDA")
    state = load_state(args.resume or args.eval_checkpoint) if args.resume or args.eval_checkpoint else None
    args.variant = args.variant or (state["model_config"]["variant"] if state else "M0")
    args.work_dir = args.work_dir or (args.resume.parent if args.resume else
                                    ROOT / "runs/mathseg_uav" / f"{args.variant}_seed{args.seed}")
    if state and args.variant != state["model_config"]["variant"]:
        raise ValueError("Requested variant differs from checkpoint")
    if args.resume and state["contract"] != contract(args, hashes):
        differences = [k for k, v in contract(args, hashes).items() if state["contract"].get(k) != v]
        raise ValueError(f"Resume protocol differs: {differences}; reuse original training arguments")
    if args.eval_checkpoint and hashes != state["contract"]["splits"]:
        raise ValueError("Evaluation split hashes differ from the training protocol")
    if not args.eval_checkpoint:
        args.work_dir.mkdir(parents=True, exist_ok=True)
        if not args.resume and any((args.work_dir / name).exists() for name in ("config.json", "last.pt", "best.pt")):
            raise FileExistsError("Run already exists; use --resume or a new --work-dir")
    seed_everything(args.seed)
    model = (MathSegUAV(**state["model_config"]) if state else MathSegUAV(
        variant=args.variant, width=args.width, model_name=args.model_name,
        pretrained=not args.no_pretrained, local_files_only=not args.allow_download,
        prior_channels=args.prior_channels, prior_mode=args.prior_mode,
    )).to(device)
    if state:
        model.load_state_dict(state["model"], strict=True)
    if args.eval_checkpoint:
        loader = make_loader(images, masks, splits[args.eval_split], workers=args.num_workers)
        metrics = evaluate(model, loader, device)
        metrics.update(checkpoint=str(args.eval_checkpoint), split=args.eval_split,
                       variant=args.variant, split_hashes=hashes, inference="native_whole_fp32_no_tta")
        output = args.eval_checkpoint.with_name(args.eval_checkpoint.stem + f"_{args.eval_split}_metrics.json")
        write_json(output, metrics)
        print(json.dumps(metrics, indent=2))
        return
    encoder_params = list(model.encoder.parameters())
    encoder_ids = {id(p) for p in encoder_params}
    optimizer = torch.optim.AdamW([
        {"params": encoder_params, "lr": args.lr * args.backbone_lr_multiplier},
        {"params": [p for p in model.parameters() if id(p) not in encoder_ids], "lr": args.lr},
    ], weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp)
    start, best, metrics = 0, -1.0, None
    counts = None
    if args.variant == "M4":
        counts = state["class_counts"] if state else scan_masks(images, masks, splits["train"])
        if not counts or sum(counts[1:]) == 0:
            raise ValueError("M4 needs valid training class counts")
        write_json(args.work_dir / "class_counts.json", {"counts": counts, "train_hash": hashes["train"]})
    if args.resume:
        optimizer.load_state_dict(state["optimizer"])
        scaler.load_state_dict(state["scaler"])
        start, best, metrics = state["update"], state["best_mIoU"], state["metrics"]
    stop = args.stop_after or args.max_updates
    if start >= stop:
        raise ValueError(f"Checkpoint is already at update {start}; requested stop is {stop}")
    loader = make_loader(images, masks, splits["train"], workers=args.num_workers, size=args.size,
                         seed=args.seed, batch_size=args.batch_size,
                         start_microbatch=start * args.accumulation,
                         total_microbatches=stop * args.accumulation)
    val_loader = make_loader(images, masks, splits["val"], workers=args.num_workers,
                             size=args.val_size or None)
    configuration = {"args": vars(args), "contract": contract(args, hashes), "model_config": model.model_config,
                     "train_samples": len(splits["train"]), "val_samples": len(splits["val"]),
                     "parameters": sum(p.numel() for p in model.parameters()),
                     "environment": {"python": platform.python_version(), "torch": torch.__version__,
                                     "transformers": transformers.__version__,
                                     "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
                                     "cuda": torch.version.cuda,
                                     "device_requested": str(device)}}
    write_json(args.work_dir / ("resume_config.json" if args.resume else "config.json"), configuration)
    write_json(args.work_dir / "status.json", {"completed": False, "update": start,
                                               "max_updates": args.max_updates, "state": "running"})
    print(json.dumps(configuration, default=str), flush=True)
    iterator = iter(loader)
    if args.resume:
        restore_rng(state["rng"])
    started = time.monotonic()
    for update in range(start + 1, stop + 1):
        model.train()
        factor = (1 - (update - 1) / args.max_updates) ** 0.9
        for group, multiplier in zip(optimizer.param_groups, (args.backbone_lr_multiplier, 1)):
            group["lr"] = args.lr * multiplier * factor
        weights = class_weights(counts, update / args.max_updates, args.gamma_max).to(device) if counts else None
        optimizer.zero_grad(set_to_none=True)
        loss_value = 0.0
        for _ in range(args.accumulation):
            batch = next(iterator)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=args.amp):
                output = model(batch["image"].to(device, non_blocking=True))
                loss = segmentation_loss(output, batch["mask"].to(device, non_blocking=True), weights) / args.accumulation
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss at update {update}")
            scaler.scale(loss).backward()
            loss_value += float(loss.detach())
        scaler.unscale_(optimizer)
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        scaler.step(optimizer)
        scaler.update()
        if update % args.log_interval == 0 or update == start + 1:
            record = {"update": update, "loss": loss_value, "lr": optimizer.param_groups[1]["lr"],
                      "grad_norm": float(norm), "seconds_per_update_including_val": (time.monotonic() - started) / (update - start)}
            if weights is not None:
                record["class_weights"] = weights.tolist()
            print(json.dumps(record), flush=True)
        if update % args.val_interval == 0 or update == stop:
            metrics = {"update": update, **evaluate(model, val_loader, device)}
            if metrics["mIoU"] is None:
                raise ValueError("Validation set contains no valid pixels")
            print(json.dumps(metrics), flush=True)
            with (args.work_dir / "metrics.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(metrics) + "\n")
            write_json(args.work_dir / "val_metrics.json", metrics)
            if metrics["mIoU"] > best:
                best = metrics["mIoU"]
                save_checkpoint(args.work_dir / "best.pt", model, optimizer, scaler, update, best, metrics, args, hashes, counts)
                write_json(args.work_dir / "best_metrics.json", metrics)
        if update % args.save_interval == 0 or update % args.val_interval == 0 or update == stop:
            save_checkpoint(args.work_dir / "last.pt", model, optimizer, scaler, update, best, metrics, args, hashes, counts)
    write_json(args.work_dir / "status.json", {"completed": stop == args.max_updates, "update": stop,
                                               "max_updates": args.max_updates, "best_mIoU": best})
    print(f"{'completed' if stop == args.max_updates else 'paused'}: {args.work_dir}", flush=True)


if __name__ == "__main__":
    main()
