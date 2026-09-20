"""Fine-tune the HyperSeg-UAV checkpoint on a few labeled LoveDA images."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset, RandomSampler


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hyperseg_uav import HyperSegUAV


CLASS_NAMES = (
    "no_data",
    "background",
    "building",
    "road",
    "water",
    "barren",
    "forest",
    "agriculture",
)
NUM_CLASSES = len(CLASS_NAMES)
IGNORE_INDEX = 0


@dataclass(frozen=True)
class Sample:
    domain: str
    name: str
    image: Path
    mask: Path

    @property
    def key(self) -> str:
        return f"{self.domain}/{self.name}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True, help="LoveDA root containing Train/ and Val/")
    parser.add_argument("--train-split", default="Train", help="Split used as the few-shot sample pool")
    parser.add_argument("--val-split", default="Val", help="Split used for evaluation")
    parser.add_argument("--init-checkpoint", type=Path, default=ROOT / "models/hyperseg_b3_best.pt")
    parser.add_argument("--head-init", choices=("random", "mapped"), default="random",
                        help="mapped copies source UAV classes 1..5 and 7 into the LoveDA head")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--manifest", type=Path, default=None, help="Reuse a selection.json from another run")
    parser.add_argument("--shots", type=int, default=5, help="Labeled training images selected per domain")
    parser.add_argument("--domains", nargs="+", choices=("urban", "rural"), default=("urban", "rural"))
    parser.add_argument("--mode", choices=("head", "adapter", "full"), default="adapter")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--steps-per-epoch", type=int, default=100)
    parser.add_argument("--crop-size", type=int, default=512)
    parser.add_argument("--eval-size", type=int, default=1024, help="Square validation size; use 0 for native size")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--eval-batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--encoder-lr-multiplier", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--val-interval", type=int, default=5)
    parser.add_argument("--restore-best", action=argparse.BooleanOptionalAction, default=True,
                        help="after a non-improving validation, restore the best full checkpoint before continuing")
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    if args.shots < 1 or args.epochs < 1 or args.steps_per_epoch < 1:
        parser.error("--shots, --epochs, and --steps-per-epoch must be positive")
    if args.crop_size < 1 or args.eval_size < 0:
        parser.error("--crop-size must be positive and --eval-size cannot be negative")
    if args.val_interval < 1:
        parser.error("--val-interval must be positive")
    return args


def resolve_child(parent: Path, wanted: str) -> Path:
    if not parent.is_dir():
        raise FileNotFoundError(f"Directory does not exist: {parent}")
    matches = [path for path in parent.iterdir() if path.is_dir() and path.name.lower() == wanted.lower()]
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one '{wanted}' directory under {parent}, found {len(matches)}")
    return matches[0]


def discover_samples(data_root: Path, split: str, domains: tuple[str, ...]) -> list[Sample]:
    split_dir = resolve_child(data_root, split)
    records: list[Sample] = []
    for requested_domain in domains:
        domain_dir = resolve_child(split_dir, requested_domain)
        image_dir = resolve_child(domain_dir, "images_png")
        mask_dir = resolve_child(domain_dir, "masks_png")
        for image in sorted(image_dir.glob("*.png")):
            mask = mask_dir / image.name
            if not mask.is_file():
                raise FileNotFoundError(f"Missing mask for {image}: {mask}")
            records.append(Sample(requested_domain.lower(), image.name, image, mask))
    if not records:
        raise RuntimeError(f"No PNG image/mask pairs found in LoveDA {split} split")
    return records


def read_mask(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        mask = np.asarray(image)
    if mask.ndim != 2:
        raise ValueError(f"Mask must be a single-channel label PNG, got shape {mask.shape}: {path}")
    mask = mask.astype(np.uint8, copy=True)
    mask[mask == 255] = IGNORE_INDEX
    invalid = np.unique(mask[mask >= NUM_CLASSES])
    if invalid.size:
        raise ValueError(f"Mask {path} contains unsupported labels: {invalid.tolist()}")
    return mask


def select_few_shot(samples: list[Sample], shots: int, seed: int) -> list[Sample]:
    """Select K images per domain while greedily balancing class coverage."""
    selected: list[Sample] = []
    for domain in sorted({sample.domain for sample in samples}):
        candidates = [sample for sample in samples if sample.domain == domain]
        if shots > len(candidates):
            raise ValueError(f"Requested {shots} shots for {domain}, but only {len(candidates)} samples exist")
        rng = random.Random(f"{seed}:{domain}")
        rng.shuffle(candidates)
        present = {
            sample.key: set(int(label) for label in np.unique(read_mask(sample.mask)) if label != IGNORE_INDEX)
            for sample in candidates
        }
        counts = {label: 0 for label in range(1, NUM_CLASSES)}
        remaining = list(candidates)
        for _ in range(shots):
            best = max(
                remaining,
                key=lambda sample: sum(1.0 / (1.0 + counts[label]) for label in present[sample.key]),
            )
            selected.append(best)
            for label in present[best.key]:
                counts[label] += 1
            remaining.remove(best)
    return sorted(selected, key=lambda sample: sample.key)


def write_manifest(path: Path, selected: list[Sample], args: argparse.Namespace) -> None:
    class_presence = {}
    for sample in selected:
        labels = [int(label) for label in np.unique(read_mask(sample.mask)) if label != IGNORE_INDEX]
        class_presence[sample.key] = labels
    payload = {
        "protocol": "K labeled images per domain, selected greedily for class coverage",
        "shots_per_domain": args.shots,
        "domains": list(args.domains),
        "seed": args.seed,
        "classes": {str(index): name for index, name in enumerate(CLASS_NAMES)},
        "ignore_index": IGNORE_INDEX,
        "samples": [sample.key for sample in selected],
        "class_presence": class_presence,
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def samples_from_manifest(path: Path, available: list[Sample]) -> list[Sample]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    keys = payload.get("samples")
    if not isinstance(keys, list) or not keys:
        raise ValueError(f"Manifest has no non-empty 'samples' list: {path}")
    by_key = {sample.key: sample for sample in available}
    missing = [key for key in keys if key not in by_key]
    if missing:
        raise ValueError(f"Manifest samples are absent from this dataset: {missing[:5]}")
    if len(keys) != len(set(keys)):
        raise ValueError(f"Manifest contains duplicate samples: {path}")
    return [by_key[key] for key in keys]


class LoveDADataset(Dataset):
    def __init__(self, samples: list[Sample], size: int, training: bool) -> None:
        self.samples = samples
        self.size = size
        self.training = training

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, object]:
        sample = self.samples[index]
        with Image.open(sample.image) as source:
            image = source.convert("RGB")
        mask = Image.fromarray(read_mask(sample.mask), mode="L")
        if self.training:
            scale = random.uniform(0.5, 1.5)
            width = max(self.size, round(image.width * scale))
            height = max(self.size, round(image.height * scale))
            image = image.resize((width, height), Image.Resampling.BILINEAR)
            mask = mask.resize((width, height), Image.Resampling.NEAREST)
            left = random.randint(0, width - self.size)
            top = random.randint(0, height - self.size)
            box = (left, top, left + self.size, top + self.size)
            image, mask = image.crop(box), mask.crop(box)
            if random.random() < 0.5:
                image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
                mask = mask.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            if random.random() < 0.5:
                image = image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
                mask = mask.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
            if random.random() < 0.7:
                image = ImageEnhance.Brightness(image).enhance(random.uniform(0.8, 1.2))
                image = ImageEnhance.Contrast(image).enhance(random.uniform(0.8, 1.2))
            if random.random() < 0.4:
                image = ImageEnhance.Color(image).enhance(random.uniform(0.8, 1.2))
        elif self.size:
            image = image.resize((self.size, self.size), Image.Resampling.BILINEAR)
            mask = mask.resize((self.size, self.size), Image.Resampling.NEAREST)

        image_array = np.asarray(image, dtype=np.float32) / 255.0
        mask_array = np.asarray(mask, dtype=np.int64).copy()
        return {
            "image": torch.from_numpy(np.ascontiguousarray(image_array)).permute(2, 0, 1),
            "mask": torch.from_numpy(mask_array),
            "name": sample.key,
        }


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def compute_class_weights(samples: list[Sample]) -> tuple[torch.Tensor, list[int]]:
    histogram = np.zeros(NUM_CLASSES, dtype=np.int64)
    for sample in samples:
        histogram += np.bincount(read_mask(sample.mask).reshape(-1), minlength=NUM_CLASSES)[:NUM_CLASSES]
    frequencies = histogram[1:] / max(1, histogram[1:].sum())
    weights = np.zeros(NUM_CLASSES, dtype=np.float32)
    present = frequencies > 0
    weights[1:][present] = 1.0 / np.log(1.02 + frequencies[present])
    if present.any():
        weights[1:][present] /= weights[1:][present].mean()
    return torch.from_numpy(weights), histogram.tolist()


def loveda_loss(
    output: dict[str, torch.Tensor],
    target: torch.Tensor,
    class_weights: torch.Tensor,
) -> torch.Tensor:
    logits = output["logits"]
    valid = target != IGNORE_INDEX
    per_pixel_ce = F.cross_entropy(
        logits, target, weight=class_weights, ignore_index=IGNORE_INDEX, reduction="none"
    )
    ce = per_pixel_ce[valid].mean() if valid.any() else logits.sum() * 0.0

    probability = logits.softmax(dim=1)
    dice_losses = []
    for label in range(1, NUM_CLASSES):
        truth = (target == label) & valid
        if truth.any():
            prediction = probability[:, label][valid]
            truth_float = truth[valid].float()
            dice_losses.append(
                1.0 - (2.0 * (prediction * truth_float).sum() + 1.0)
                / (prediction.sum() + truth_float.sum() + 1.0)
            )
    dice = torch.stack(dice_losses).mean() if dice_losses else logits.sum() * 0.0

    boundary_target = torch.zeros_like(target, dtype=torch.bool)
    boundary_valid = torch.zeros_like(target, dtype=torch.bool)
    vertical_valid = valid[:, 1:] & valid[:, :-1]
    horizontal_valid = valid[:, :, 1:] & valid[:, :, :-1]
    boundary_target[:, 1:] |= (target[:, 1:] != target[:, :-1]) & vertical_valid
    boundary_target[:, :, 1:] |= (target[:, :, 1:] != target[:, :, :-1]) & horizontal_valid
    boundary_valid[:, 1:] |= vertical_valid
    boundary_valid[:, :, 1:] |= horizontal_valid
    boundary_bce = F.binary_cross_entropy_with_logits(
        output["boundary"].squeeze(1), boundary_target.float(), reduction="none"
    )
    boundary = boundary_bce[boundary_valid].mean() if boundary_valid.any() else logits.sum() * 0.0
    return torch.nan_to_num(0.7 * ce + 0.2 * dice + 0.1 * boundary, nan=0.0)


def legacy_encoder_key(key: str) -> str | None:
    """Map Transformers 5 SegFormer keys to their Transformers 4 names."""
    prefix = "encoder.backbone.stages."
    if not key.startswith(prefix):
        return None
    stage_text, separator, suffix = key[len(prefix):].partition(".")
    if not separator or not stage_text.isdigit():
        return None
    if suffix.startswith("patch_embeddings."):
        suffix = suffix.removeprefix("patch_embeddings.")
        return f"encoder.backbone.encoder.patch_embeddings.{stage_text}.{suffix}"
    if suffix.startswith("layer_norm."):
        suffix = suffix.removeprefix("layer_norm.")
        return f"encoder.backbone.encoder.layer_norm.{stage_text}.{suffix}"
    if not suffix.startswith("blocks."):
        return None
    block_text, separator, block_suffix = suffix.removeprefix("blocks.").partition(".")
    if not separator or not block_text.isdigit():
        return None
    replacements = (
        ("layernorm_before.", "layer_norm_1."),
        ("layernorm_after.", "layer_norm_2."),
        ("attention.q_proj.", "attention.self.query."),
        ("attention.k_proj.", "attention.self.key."),
        ("attention.v_proj.", "attention.self.value."),
        (
            "attention.sequence_reduction.sequence_reduction.",
            "attention.self.sr.",
        ),
        ("attention.sequence_reduction.layer_norm.", "attention.self.layer_norm."),
        ("attention.o_proj.", "attention.output.dense."),
        ("mlp.fc1.", "mlp.dense1."),
        ("mlp.fc2.", "mlp.dense2."),
    )
    for current, legacy in replacements:
        if block_suffix.startswith(current):
            block_suffix = legacy + block_suffix.removeprefix(current)
            break
    return f"encoder.backbone.encoder.block.{stage_text}.{block_text}.{block_suffix}"


def load_transfer_checkpoint(model: nn.Module, checkpoint: dict[str, object], head_init: str = "random") -> dict[str, object]:
    if head_init not in ("random", "mapped"):
        raise ValueError(f"Unsupported head initialization: {head_init}")
    source = checkpoint.get("model", checkpoint)
    destination = model.state_dict()
    mapping = {i: i for i in (1, 2, 3, 4, 5, 7)} if head_init == "mapped" else {}
    if mapping:
        for key in ("head.weight", "head.bias"):
            if key not in source or key not in destination:
                raise ValueError(f"Mapped initialization requires {key}")
            if (source[key].shape[0] != 9 or destination[key].shape[0] != 8
                    or source[key].shape[1:] != destination[key].shape[1:]):
                raise ValueError(f"Mapped initialization requires a compatible 9-class UAV to 8-class LoveDA head: {key}")
    compatible = {}
    translated = {}
    skipped = []
    for source_key, value in source.items():
        if source_key in ("head.weight", "head.bias"):
            skipped.append(source_key)
            continue
        destination_key = source_key
        if destination_key not in destination:
            destination_key = legacy_encoder_key(source_key)
        if (
            destination_key is None
            or destination_key not in destination
            or destination[destination_key].shape != value.shape
        ):
            skipped.append(source_key)
            continue
        compatible[destination_key] = value
        if destination_key != source_key:
            translated[source_key] = destination_key
    incompatible = model.load_state_dict(compatible, strict=False)
    with torch.no_grad():
        for target_id, source_id in mapping.items():
            model.head.weight[target_id].copy_(source["head.weight"][source_id])
            model.head.bias[target_id].copy_(source["head.bias"][source_id])
    return {
        "head_init": head_init,
        "head_mapping_target_to_source": mapping,
        "head_random_channels": [i for i in range(NUM_CLASSES) if i not in mapping],
        "loaded_tensors": len(compatible),
        "translated_tensors": len(translated),
        "skipped": sorted(skipped),
        "missing": sorted(incompatible.missing_keys),
        "source_epoch": checkpoint.get("epoch"),
        "source_val_miou": checkpoint.get("val_miou"),
    }


def configure_trainable(model: HyperSegUAV, mode: str) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = False
    if mode == "head":
        modules = (model.head, model.boundary)
    elif mode == "adapter":
        modules = (
            model.scene,
            model.modulation,
            model.fusion,
            model.low_rank,
            model.decoder,
            model.head,
            model.boundary,
        )
    else:
        modules = (model,)
    for module in modules:
        for parameter in module.parameters():
            parameter.requires_grad = True


def keep_frozen_modules_in_eval(model: nn.Module) -> None:
    for module in model.modules():
        parameters = list(module.parameters())
        if parameters and not any(parameter.requires_grad for parameter in parameters):
            module.eval()


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, object]:
    confusion = torch.zeros(NUM_CLASSES, NUM_CLASSES, dtype=torch.int64, device=device)
    model.eval()
    for batch in loader:
        image = batch["image"].to(device, non_blocking=True)
        target = batch["mask"].to(device, non_blocking=True)
        prediction = model(image)["logits"].argmax(dim=1)
        valid = target != IGNORE_INDEX
        indices = target[valid] * NUM_CLASSES + prediction[valid]
        confusion += torch.bincount(indices, minlength=NUM_CLASSES**2).reshape(NUM_CLASSES, NUM_CLASSES)
    intersection = confusion.diag().float()
    union = confusion.sum(0).float() + confusion.sum(1).float() - intersection
    valid_classes = union[1:] > 0
    class_iou = intersection[1:] / union[1:].clamp_min(1)
    miou = class_iou[valid_classes].mean().item() if valid_classes.any() else 0.0
    return {
        "mIoU": miou,
        "per_class_iou": {
            CLASS_NAMES[label]: (class_iou[label - 1].item() if union[label] > 0 else None)
            for label in range(1, NUM_CLASSES)
        },
        "confusion": confusion.cpu().tolist(),
    }


def json_args(args: argparse.Namespace) -> dict[str, object]:
    return {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or (
        ROOT / "runs/loveda_fewshot" / f"{args.mode}_{args.shots}shot_seed{args.seed}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    domains = tuple(args.domains)
    train_samples = discover_samples(args.data_root, args.train_split, domains)
    val_samples = discover_samples(args.data_root, args.val_split, domains)
    selected = (
        samples_from_manifest(args.manifest, train_samples)
        if args.manifest else select_few_shot(train_samples, args.shots, args.seed)
    )
    if args.train_split.lower() == args.val_split.lower():
        selected_keys = {sample.key for sample in selected}
        val_samples = [sample for sample in val_samples if sample.key not in selected_keys]
        if not val_samples:
            raise RuntimeError("No evaluation samples remain after excluding the few-shot support set")
    selection_path = output_dir / "selection.json"
    write_manifest(selection_path, selected, args)
    class_weights, class_histogram = compute_class_weights(selected)

    train_set = LoveDADataset(selected, args.crop_size, training=True)
    val_set = LoveDADataset(val_samples, args.eval_size, training=False)
    generator = torch.Generator().manual_seed(args.seed)
    sampler = RandomSampler(
        train_set,
        replacement=True,
        num_samples=args.steps_per_epoch * args.batch_size,
        generator=generator,
    )
    common_loader_args = {
        "num_workers": args.num_workers,
        "pin_memory": torch.cuda.is_available(),
        "worker_init_fn": seed_worker,
        "persistent_workers": args.num_workers > 0,
    }
    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, sampler=sampler, drop_last=True, **common_loader_args
    )
    val_loader = DataLoader(
        val_set, batch_size=args.eval_batch_size, shuffle=False, **common_loader_args
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    source = torch.load(args.init_checkpoint, map_location="cpu", weights_only=False)
    model_config = dict(source.get("model_config", {}))
    model_config.update({"classes": NUM_CLASSES, "pretrained": False})
    model = HyperSegUAV(**model_config)
    transfer_report = load_transfer_checkpoint(model, source, args.head_init)
    del source
    configure_trainable(model, args.mode)
    model.to(device)
    class_weights = class_weights.to(device)

    encoder_parameters = [parameter for parameter in model.encoder.parameters() if parameter.requires_grad]
    encoder_ids = {id(parameter) for parameter in encoder_parameters}
    other_parameters = [
        parameter for parameter in model.parameters()
        if parameter.requires_grad and id(parameter) not in encoder_ids
    ]
    parameter_groups = []
    if encoder_parameters:
        parameter_groups.append({"params": encoder_parameters, "lr": args.lr * args.encoder_lr_multiplier})
    if other_parameters:
        parameter_groups.append({"params": other_parameters, "lr": args.lr})
    optimizer = torch.optim.AdamW(parameter_groups, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda epoch: math.pow(max(0.0, 1.0 - epoch / args.epochs), 0.9)
    )
    amp_enabled = args.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    run_info = {
        "args": json_args(args),
        "device": str(device),
        "train_samples": len(selected),
        "val_samples": len(val_samples),
        "class_histogram": dict(zip(CLASS_NAMES, class_histogram)),
        "class_weights": dict(zip(CLASS_NAMES, class_weights.detach().cpu().tolist())),
        "transfer": transfer_report,
        "trainable_parameters": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
        "total_parameters": sum(parameter.numel() for parameter in model.parameters()),
    }
    (output_dir / "run_config.json").write_text(json.dumps(run_info, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(run_info, indent=2), flush=True)

    best_miou = -1.0
    best_metrics = None
    restored_best = False
    history_path = output_dir / "history.jsonl"
    with history_path.open("w", encoding="utf-8") as history_file:
        for epoch in range(1, args.epochs + 1):
            model.train()
            keep_frozen_modules_in_eval(model)
            loss_sum = 0.0
            update_count = 0
            for batch in train_loader:
                image = batch["image"].to(device, non_blocking=True)
                target = batch["mask"].to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", enabled=amp_enabled):
                    loss = loveda_loss(model(image), target, class_weights)
                if not torch.isfinite(loss):
                    continue
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    (parameter for parameter in model.parameters() if parameter.requires_grad), 1.0
                )
                scaler.step(optimizer)
                scaler.update()
                loss_sum += loss.detach().item()
                update_count += 1
            scheduler.step()

            should_validate = epoch % args.val_interval == 0 or epoch == args.epochs
            row: dict[str, object] = {
                "epoch": epoch,
                "train_loss": loss_sum / max(1, update_count),
                "optimizer_updates": update_count,
                "lr": [group["lr"] for group in optimizer.param_groups],
            }
            if should_validate:
                metrics = evaluate(model, val_loader, device)
                row.update(metrics)
                if metrics["mIoU"] > best_miou:
                    best_miou = float(metrics["mIoU"])
                    best_metrics = metrics
                    torch.save(
                        {
                            "model": model.state_dict(),
                            "model_config": model.model_config,
                            "epoch": epoch,
                            "val_miou": best_miou,
                            "experiment": run_info,
                            "optimizer": optimizer.state_dict(),
                            "scheduler": scheduler.state_dict(),
                            "scaler": scaler.state_dict(),
                        },
                        output_dir / "best.pt",
                    )
                    restored_best = False
                elif args.restore_best and best_miou >= 0:
                    best_checkpoint = torch.load(output_dir / "best.pt", map_location=device, weights_only=False)
                    model.load_state_dict(best_checkpoint["model"])
                    if "optimizer" in best_checkpoint:
                        optimizer.load_state_dict(best_checkpoint["optimizer"])
                    if "scheduler" in best_checkpoint:
                        scheduler.load_state_dict(best_checkpoint["scheduler"])
                    if "scaler" in best_checkpoint:
                        scaler.load_state_dict(best_checkpoint["scaler"])
                    restored_best = True
                else:
                    restored_best = False
            row["restored_best"] = restored_best if should_validate else False
            history_file.write(json.dumps(row) + "\n")
            history_file.flush()
            metric_text = f", val_mIoU={row['mIoU']:.4f}" if "mIoU" in row else ""
            restore_text = ", restored_best=true" if row["restored_best"] else ""
            print(
                f"epoch {epoch}/{args.epochs}: train_loss={row['train_loss']:.4f}{metric_text}{restore_text}",
                flush=True,
            )

    summary = {
        "mode": args.mode,
        "shots_per_domain": args.shots,
        "domains": list(domains),
        "seed": args.seed,
        "best_mIoU": best_miou,
        "best_metrics": best_metrics,
        "checkpoint": str(output_dir / "best.pt"),
        "selection": str(selection_path),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
