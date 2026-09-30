"""Train H3 (Swin-L + HyperSeg) with the background-collapse fix.

Control-arm design: every knob that the fix touches is independent, and the presets only
pick defaults, so a single-variable attribution run is possible without editing code.

| preset   | augmentation | false-bg penalty | class weights | selection        |
|----------|--------------|------------------|---------------|------------------|
| baseline | legacy       | 0.00             | none          | native mIoU      |
| loss     | legacy       | 0.25             | mild          | native mIoU      |
| aug      | bgfix        | 0.00             | none          | native mIoU      |
| full     | bgfix        | 0.25             | mild          | native + proxy   |

``aug=legacy`` reuses ``hyperseg_uav.UAVDataset`` untouched, so the baseline arm cannot
drift from the published H3 result through a re-implementation of the loader.

Validation always reports the whole-split, native-resolution confusion matrix (the
protocol the published H3 checkpoint was selected under).  On top of that it optionally
reports a *proxy* mIoU: the same labelled val images rendered at a reduced detail scale
(0.5x / 0.625x / 0.75x), which reproduces the one axis of the test-set shift that can be
built from labelled data.  The proxy is a necessary, not sufficient, stand-in -- it does
not model the colour cast, the haze or the new scene types.

Selection score: ``(1 - w) * native_mIoU + w * proxy_mIoU`` with ``w = --proxy-weight``
(default 0.3), so native quality stays the primary criterion while the proxy guards
against a checkpoint that only looks good at full detail scale.
"""

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
HERE = Path(__file__).resolve().parent
for extra in (str(ROOT), str(HERE)):
    if extra not in sys.path:
        sys.path.insert(0, extra)

from hyperseg_uav import UAVDataset  # noqa: E402
from hyperseg_uav.swin_l_model import SwinLHyperSeg  # noqa: E402
from bgfix_dataset import BgFixUAVDataset, worker_init_fn  # noqa: E402
from bgfix_loss import bgfix_loss, confusion_matrix, inverse_frequency_weights, metrics_from_confusion  # noqa: E402

DEFAULT_BACKBONE = (
    "https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/swin/"
    "swin_large_patch4_window12_384_22k_20220412-6580f57d.pth"
)

PRESETS = {
    "baseline": {"aug": "legacy", "false_bg": 0.0, "class_weight": "none", "proxy_weight": 0.0},
    "loss": {"aug": "legacy", "false_bg": 0.25, "class_weight": "mild", "proxy_weight": 0.0},
    "aug": {"aug": "bgfix", "false_bg": 0.0, "class_weight": "none", "proxy_weight": 0.0},
    "full": {"aug": "bgfix", "false_bg": 0.25, "class_weight": "mild", "proxy_weight": 0.30},
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--preset", choices=tuple(PRESETS), default="full")

    parser.add_argument("--data-root", type=Path, default=ROOT / "dataset/low_altitude_2026")
    parser.add_argument("--image-dir", type=Path)
    parser.add_argument("--mask-dir", type=Path)
    parser.add_argument("--split-dir", type=Path, default=ROOT / "runs/splits")
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--class-counts", type=Path,
                        default=HERE / "configs/train_class_pixel_counts.json")
    parser.add_argument("--backbone-checkpoint", default=DEFAULT_BACKBONE)
    parser.add_argument("--no-pretrained", action="store_true")

    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--accumulative-counts", type=int, default=1)
    parser.add_argument("--max-iters", type=int, default=160000)
    parser.add_argument("--val-interval", type=int, default=2800)
    parser.add_argument("--val-size", type=int, default=0)
    parser.add_argument("--val-max-images", type=int, default=0, help="0 keeps every val image")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--backbone-lr-multiplier", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--with-checkpointing", action="store_true")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--drop-optimizer-state", action="store_true",
                        help="write model weights only.  Swin-L keeps 1.47 GiB of AdamW state "
                             "against 0.74 GiB of weights, so screening runs that will not be "
                             "resumed can cut their footprint by two thirds.")
    parser.add_argument("--eval-checkpoint", type=Path)
    parser.add_argument("--eval-split", choices=("val", "test"), default="test")

    # ---- independent fix knobs (None -> take the preset default) ----
    parser.add_argument("--aug-mode", choices=("legacy", "bgfix"))
    parser.add_argument("--false-bg-weight", type=float)
    parser.add_argument("--class-weight-mode", choices=("none", "mild", "median"))
    parser.add_argument("--class-weight-floor", type=float, default=0.7)
    parser.add_argument("--class-weight-cap", type=float, default=3.0)
    parser.add_argument("--loss-rare-classes", default="5,7,8")
    parser.add_argument("--crop-rare-classes", default="3,4,5,7,8")
    parser.add_argument("--proxy-weight", type=float)
    parser.add_argument("--proxy-scales", default="0.5,0.625,0.75")
    parser.add_argument("--proxy-max-images", type=int, default=200)
    parser.add_argument("--proxy-every", type=int, default=2, help="run proxy eval every N validations")

    # ---- bgfix augmentation knobs ----
    parser.add_argument("--scale-min", type=float, default=0.4)
    parser.add_argument("--scale-max", type=float, default=1.5)
    parser.add_argument("--gsd-prob", type=float, default=0.5)
    parser.add_argument("--gsd-min", type=float, default=0.5)
    parser.add_argument("--gsd-max", type=float, default=1.0)
    parser.add_argument("--photo-prob", type=float, default=0.8)
    parser.add_argument("--scene-crop-prob", type=float, default=0.5)
    return parser.parse_args()


def resolve_settings(args):
    preset = PRESETS[args.preset]
    args.aug_mode = args.aug_mode or preset["aug"]
    if args.false_bg_weight is None:
        args.false_bg_weight = preset["false_bg"]
    args.class_weight_mode = args.class_weight_mode or preset["class_weight"]
    if args.proxy_weight is None:
        args.proxy_weight = preset["proxy_weight"]
    args.loss_rare_classes = tuple(int(v) for v in str(args.loss_rare_classes).split(",") if v.strip())
    args.crop_rare_classes = tuple(int(v) for v in str(args.crop_rare_classes).split(",") if v.strip())
    args.proxy_scales = tuple(float(v) for v in str(args.proxy_scales).split(",") if v.strip())
    return args


def resolve_dirs(data_root: Path, image_dir, mask_dir):
    if image_dir and mask_dir:
        return image_dir, mask_dir
    for candidate_image, candidate_mask in (
        (data_root / "train" / "images", data_root / "train" / "masks"),
        (data_root / "train" / "train" / "images", data_root / "train" / "train" / "masks"),
        (data_root / "low_altitude_2026" / "train" / "images",
         data_root / "low_altitude_2026" / "train" / "masks"),
    ):
        if candidate_image.is_dir() and candidate_mask.is_dir():
            return candidate_image, candidate_mask
    raise FileNotFoundError("Could not find labelled data; pass --image-dir and --mask-dir.")


def read_ids(split_dir: Path, name: str, limit: int = 0):
    path = split_dir / f"{name}.txt"
    values = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not values:
        raise ValueError(f"Empty split: {path}")
    if 0 < limit < len(values):
        rng = random.Random(3407)
        values = sorted(rng.sample(values, limit))
    return values


def build_dataset(args, image_dir, mask_dir, sample_ids, size, training, eval_downscale=None):
    """Build the training or evaluation dataset for the active arm.

    The evaluation path is deterministic for every arm, so a reduced-detail proxy set is
    always built from ``BgFixUAVDataset`` regardless of ``--aug-mode``; only the training
    path differs between the legacy and bgfix arms.
    """
    if training:
        if args.aug_mode == "legacy":
            return UAVDataset(image_dir, mask_dir, sample_ids, size, True,
                              scene_crop_prob=0.3, rare_classes=(5, 7, 8))
        return BgFixUAVDataset(
            image_dir, mask_dir, sample_ids, size, True,
            scale_range=(args.scale_min, args.scale_max),
            enable_gsd=args.gsd_prob > 0, gsd_prob=args.gsd_prob,
            gsd_range=(args.gsd_min, args.gsd_max),
            enable_photometric=args.photo_prob > 0, photo_prob=args.photo_prob,
            scene_crop_prob=args.scene_crop_prob,
            rare_classes=args.crop_rare_classes,
        )
    if eval_downscale is not None:
        return BgFixUAVDataset(image_dir, mask_dir, sample_ids, None, False,
                               eval_downscale=eval_downscale)
    if args.aug_mode == "legacy":
        return UAVDataset(image_dir, mask_dir, sample_ids, size, False)
    return BgFixUAVDataset(image_dir, mask_dir, sample_ids, size, False)


def make_loader(dataset, training, batch_size, workers):
    if not len(dataset):
        raise ValueError("Empty dataset after applying the split ids")
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=training,
        drop_last=training,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=workers > 0,
        worker_init_fn=worker_init_fn if training else None,
    )


@torch.no_grad()
def evaluate(model, loader, device, classes: int = 9):
    confusion = torch.zeros(classes, classes, dtype=torch.int64, device=device)
    model.eval()
    for batch in loader:
        logits = model(batch["image"].to(device, non_blocking=True))["logits"]
        confusion += confusion_matrix(logits, batch["mask"].to(device, non_blocking=True), classes=classes)
    return metrics_from_confusion(confusion)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def set_poly_lr(optimizer, base_lr, backbone_multiplier, step, max_iters, power):
    factor = max(0.0, (1.0 - step / max_iters)) ** power
    optimizer.param_groups[0]["lr"] = base_lr * backbone_multiplier * factor
    optimizer.param_groups[1]["lr"] = base_lr * factor


def save_checkpoint(path, model, optimizer, step, best, args, metrics, class_weights,
                    keep_optimizer=True):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    torch.save({
        "model": model.state_dict(),
        "model_config": model.model_config,
        "optimizer": optimizer.state_dict() if keep_optimizer else None,
        "step": step,
        "best_mIoU": best,
        "metrics": metrics,
        "experiment": "H3_swin_l_hyperseg_bgfix",
        "args": {key: (str(value) if isinstance(value, Path) else value) for key, value in vars(args).items()},
        "class_weights": None if class_weights is None else class_weights.tolist(),
    }, temporary)
    temporary.replace(path)


def main():
    args = resolve_settings(parse_args())
    if args.batch_size < 1 or args.size < 1 or args.max_iters < 1 or args.val_interval < 1:
        raise SystemExit("invalid batch/size/iteration arguments")
    set_seed(args.seed)
    image_dir, mask_dir = resolve_dirs(args.data_root, args.image_dir, args.mask_dir)

    splits = {name: set(read_ids(args.split_dir, name)) for name in ("train", "val", "test")}
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        if splits[left] & splits[right]:
            raise ValueError(f"split overlap between {left} and {right}")
    for path in sorted(Path(image_dir).glob("*.png"))[:1]:
        if not (Path(mask_dir) / path.name).is_file():
            raise FileNotFoundError(Path(mask_dir) / path.name)

    # ---- class weights -------------------------------------------------------
    class_weights = None
    counts_payload = None
    if args.class_weight_mode != "none":
        payload = json.loads(Path(args.class_counts).read_text(encoding="utf-8"))
        counts = [int(payload["class_pixel_counts"][str(cls)]) for cls in range(9)]
        class_weights = inverse_frequency_weights(
            counts, mode=args.class_weight_mode,
            floor=args.class_weight_floor, cap=args.class_weight_cap,
        )
        counts_payload = {**payload, "mode": args.class_weight_mode,
                          "resolved_weights": class_weights.tolist()}
        print("class weights:", class_weights.tolist(), flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = None if args.no_pretrained or args.resume or args.eval_checkpoint else args.backbone_checkpoint
    model = SwinLHyperSeg(pretrained_checkpoint=checkpoint, with_cp=args.with_checkpointing).to(device)

    if args.eval_checkpoint:
        state = torch.load(args.eval_checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        loader = make_loader(
            build_dataset(args, image_dir, mask_dir, read_ids(args.split_dir, args.eval_split),
                          args.val_size or None, False),
            False, 1, args.num_workers)
        metrics = evaluate(model, loader, device)
        metrics["checkpoint"] = str(args.eval_checkpoint)
        metrics["split"] = args.eval_split
        print(json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)
        output = args.eval_checkpoint.with_name(args.eval_checkpoint.stem + f"_{args.eval_split}_metrics.json")
        output.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"metrics written to: {output}", flush=True)
        return

    train_dataset = build_dataset(args, image_dir, mask_dir, read_ids(args.split_dir, "train"),
                                  args.size, True)
    val_ids = read_ids(args.split_dir, "val", limit=args.val_max_images)
    val_dataset = build_dataset(args, image_dir, mask_dir, val_ids, args.val_size or None, False)
    proxy_datasets = {
        scale: build_dataset(args, image_dir, mask_dir, read_ids(args.split_dir, "val", limit=args.proxy_max_images),
                             None, False, eval_downscale=scale)
        for scale in args.proxy_scales
    } if args.proxy_weight > 0 else {}

    train_loader = make_loader(train_dataset, True, args.batch_size, args.num_workers)
    val_loader = make_loader(val_dataset, False, 1, args.num_workers)
    proxy_loaders = {scale: make_loader(ds, False, 1, args.num_workers) for scale, ds in proxy_datasets.items()}

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
        if state.get("optimizer") is None:
            raise SystemExit(
                f"{args.resume} was written with --drop-optimizer-state and cannot resume. "
                "Re-run it from scratch, or resume from a checkpoint that kept the optimizer."
            )
        optimizer.load_state_dict(state["optimizer"])
        start_step, best = int(state.get("step", 0)), float(state.get("best_mIoU", -1.0))

    args.work_dir.mkdir(parents=True, exist_ok=True)
    configuration = {
        **{key: (str(value) if isinstance(value, Path) else value) for key, value in vars(args).items()},
        "image_dir": str(image_dir), "mask_dir": str(mask_dir),
        "model_config": model.model_config,
        "train_samples": len(train_dataset), "val_samples": len(val_dataset),
        "proxy_val_samples": {str(scale): len(ds) for scale, ds in proxy_datasets.items()},
        "class_weight_payload": counts_payload,
        "param_counts": {
            "total": sum(p.numel() for p in model.parameters()),
            "backbone": sum(p.numel() for p in backbone_params),
        },
    }
    (args.work_dir / "config.json").write_text(
        json.dumps(configuration, default=str, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(configuration, default=str, ensure_ascii=False), flush=True)

    started = time.monotonic()
    iterator = iter(train_loader)
    weights_on_device = None if class_weights is None else class_weights.to(device)
    validation_count = 0
    for step in range(start_step, args.max_iters):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss_value, components = 0.0, {}
        for _ in range(args.accumulative_counts):
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(train_loader)
                batch = next(iterator)
            total, components = bgfix_loss(
                model(batch["image"].to(device, non_blocking=True)),
                batch["mask"].to(device, non_blocking=True),
                class_weights=weights_on_device,
                rare_classes=args.loss_rare_classes,
                false_bg_weight=args.false_bg_weight,
                return_components=True,
            )
            loss = total / args.accumulative_counts
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
                  f"false_bg={components.get('loss/false_bg', 0.0):.4f} "
                  f"grad_norm={float(grad_norm):.4f} "
                  f"sec_per_update={(time.monotonic() - started) / (update - start_step):.3f} "
                  f"peak_gpu_gb={(torch.cuda.max_memory_allocated() / 1e9) if device.type == 'cuda' else 0:.2f}",
                  flush=True)

        if update % args.val_interval == 0 or update == args.max_iters:
            validation_count += 1
            record = {"iter": update}
            native = evaluate(model, val_loader, device)
            record["val"] = native
            proxy_summary = {}
            if proxy_loaders and (validation_count % max(1, args.proxy_every) == 0):
                for scale, loader in proxy_loaders.items():
                    proxy_summary[str(scale)] = evaluate(model, loader, device)
                record["proxy"] = proxy_summary
            if proxy_summary:
                proxy_miou = float(np.mean([value["mIoU"] for value in proxy_summary.values()]))
                record["proxy_mIoU_mean"] = proxy_miou
                record["selection_score"] = (
                    (1.0 - args.proxy_weight) * native["mIoU"] + args.proxy_weight * proxy_miou
                )
            else:
                record["selection_score"] = native["mIoU"]
            print(json.dumps(record, ensure_ascii=False), flush=True)
            (args.work_dir / "val_metrics.json").write_text(
                json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            history = args.work_dir / "val_history.jsonl"
            with history.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            if record["selection_score"] > best:
                best = record["selection_score"]
                save_checkpoint(args.work_dir / "best.pt", model, optimizer, update, best, args, record,
                                class_weights, keep_optimizer=not args.drop_optimizer_state)
            save_checkpoint(args.work_dir / "last.pt", model, optimizer, update, best, args, record,
                            class_weights, keep_optimizer=not args.drop_optimizer_state)
    print(f"completed: {args.work_dir}", flush=True)


if __name__ == "__main__":
    main()
