"""Train and evaluate one fixed-protocol SUIM few-shot run."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiment.fewShot_SUIM.constants import CLASS_NAMES, NUM_CLASSES
from experiment.fewShot_SUIM.protocol import (
    SHOTS, digest, load_protocol, resolve_protocol_paths, resolve_protocol_samples, write_json,
)


SEMANTIC_MAPPING = {0: 4, 2: 6, 3: 2, 7: 5}


def legacy_encoder_key(key: str) -> str | None:
    """Map Transformers 5 SegFormer state keys to their Transformers 4 names."""
    prefix = "encoder.backbone.stages."
    if not key.startswith(prefix):
        return None
    stage_text, separator, suffix = key[len(prefix):].partition(".")
    if not separator or not stage_text.isdigit():
        return None
    if suffix.startswith("patch_embeddings."):
        return f"encoder.backbone.encoder.patch_embeddings.{stage_text}.{suffix.removeprefix('patch_embeddings.')}"
    if suffix.startswith("layer_norm."):
        return f"encoder.backbone.encoder.layer_norm.{stage_text}.{suffix.removeprefix('layer_norm.')}"
    if not suffix.startswith("blocks."):
        return None
    block_text, separator, block_suffix = suffix.removeprefix("blocks.").partition(".")
    if not separator or not block_text.isdigit():
        return None
    replacements = (
        ("layernorm_before.", "layer_norm_1."), ("layernorm_after.", "layer_norm_2."),
        ("attention.q_proj.", "attention.self.query."),
        ("attention.k_proj.", "attention.self.key."),
        ("attention.v_proj.", "attention.self.value."),
        ("attention.sequence_reduction.sequence_reduction.", "attention.self.sr."),
        ("attention.sequence_reduction.layer_norm.", "attention.self.layer_norm."),
        ("attention.o_proj.", "attention.output.dense."),
        ("mlp.fc1.", "mlp.dense1."), ("mlp.fc2.", "mlp.dense2."),
    )
    for current, legacy in replacements:
        if block_suffix.startswith(current):
            block_suffix = legacy + block_suffix.removeprefix(current)
            break
    return f"encoder.backbone.encoder.block.{stage_text}.{block_text}.{block_suffix}"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=ROOT / "SUIM")
    parser.add_argument("--manifest-dir", type=Path, required=True)
    parser.add_argument("--init-checkpoint", type=Path, default=ROOT / "models/hyperseg_resume_best.pt")
    parser.add_argument("--checkpoint-sha256", help=argparse.SUPPRESS)
    parser.add_argument("--backbone-path", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--shots", type=int, choices=SHOTS, required=True)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--mode", choices=("semantic-head", "adapter", "full"), default="adapter")
    parser.add_argument("--steps", type=int, choices=(200, 2000), default=2000)
    parser.add_argument("--head-init", choices=("random", "semantic-map"), default="random")
    parser.add_argument("--crop-size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--encoder-lr-multiplier", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--skip-completed", action="store_true")
    args = parser.parse_args()
    if min(args.crop_size, args.batch_size, args.steps) < 1 or args.num_workers < 0:
        parser.error("Sizes and steps must be positive; workers cannot be negative")
    if args.lr <= 0 or args.encoder_lr_multiplier <= 0 or args.weight_decay < 0:
        parser.error("Optimizer settings are invalid")
    return args


def file_sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def load_v2(checkpoint, backbone_path=None, head_init="random"):
    import torch
    from tools.model_1 import HyperSegUAV

    config = dict(checkpoint["model_config"])
    if config.pop("version", None) != "v2" or config.get("classes") != 9:
        raise ValueError("Expected an original 9-class HyperSeg-UAV v2 checkpoint")
    config.update(classes=NUM_CLASSES, pretrained=False)
    local = ROOT / "models" / config.get("model_name", "nvidia/mit-b3").replace("/", "--")
    if backbone_path or local.is_dir():
        config["model_name"] = str(backbone_path or local)
    model = HyperSegUAV(**config)
    destination = model.state_dict()
    loaded = {}
    for source_key, tensor in checkpoint["model"].items():
        if source_key in ("head.weight", "head.bias"):
            continue
        target = source_key if source_key in destination else legacy_encoder_key(source_key)
        if target is None or target not in destination:
            raise ValueError(f"Unexpected checkpoint tensor: {source_key}")
        if target in loaded:
            raise ValueError(f"Multiple checkpoint tensors map to {target}")
        if tensor.shape != destination[target].shape:
            raise ValueError(f"Shape mismatch for {target}: {tensor.shape} vs {destination[target].shape}")
        loaded[target] = tensor
    required = set(destination) - {"head.weight", "head.bias"}
    if set(loaded) != required:
        raise ValueError(f"Incomplete non-head transfer: missing={sorted(required - set(loaded))}")
    incompatible = model.load_state_dict(loaded, strict=False)
    if set(incompatible.missing_keys) != {"head.weight", "head.bias"} or incompatible.unexpected_keys:
        raise ValueError(f"Unexpected transfer result: {incompatible}")

    mapping = SEMANTIC_MAPPING if head_init == "semantic-map" else {}
    source = checkpoint["model"]
    with torch.no_grad():
        for target_id, source_id in mapping.items():
            model.head.weight[target_id].copy_(source["head.weight"][source_id])
            model.head.bias[target_id].copy_(source["head.bias"][source_id])
    return model, {
        "head_init": head_init,
        "head_mapping_target_to_source": {str(k): v for k, v in mapping.items()},
        "head_random_channels": [c for c in range(NUM_CLASSES) if c not in mapping],
        "loaded_non_head_tensors": len(loaded),
    }


def configure_model(model, mode: str) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = False
    if mode == "semantic-head":
        modules = (model.head,)
    elif mode == "adapter":
        modules = (
            model.scene, model.modulation, model.fusion, model.low_rank,
            model.decoder, model.boundary_refine, model.head, model.boundary,
        )
    elif mode == "full":
        modules = (model,)
    else:
        raise ValueError(f"Unsupported training mode: {mode}")
    for module in modules:
        for parameter in module.parameters():
            parameter.requires_grad = True


def training_mode(model) -> None:
    from torch import nn

    model.train()
    for module in model.modules():
        parameters = list(module.parameters())
        if parameters and not any(parameter.requires_grad for parameter in parameters):
            module.eval()
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.eval()


def compute_class_weights(samples):
    import numpy as np
    import torch
    from experiment.fewShot_SUIM.data import read_sample

    histogram = np.zeros(NUM_CLASSES, dtype=np.int64)
    for sample in samples:
        _, mask, _ = read_sample(sample)
        histogram += np.bincount(mask.reshape(-1), minlength=NUM_CLASSES)[:NUM_CLASSES]
    frequencies = histogram / max(1, histogram.sum())
    if np.any(frequencies == 0):
        missing = [CLASS_NAMES[c] for c in np.flatnonzero(frequencies == 0)]
        raise ValueError(f"Few-shot support contains no pixels for classes: {missing}")
    weights = 1.0 / np.log(1.02 + frequencies)
    weights /= weights.mean()
    return torch.from_numpy(weights.astype(np.float32)), histogram.tolist()


def suim_loss(output, target, class_weights):
    import torch
    import torch.nn.functional as F

    logits = output["logits"]
    ce = F.cross_entropy(logits, target, weight=class_weights)
    probability = logits.softmax(dim=1)
    dice_losses = []
    for label in range(NUM_CLASSES):
        truth = target == label
        if truth.any():
            prediction = probability[:, label]
            truth_float = truth.float()
            dice_losses.append(
                1.0 - (2.0 * (prediction * truth_float).sum() + 1.0)
                / (prediction.sum() + truth_float.sum() + 1.0)
            )
    dice = torch.stack(dice_losses).mean()
    boundary_target = torch.zeros_like(target, dtype=torch.bool)
    boundary_target[:, 1:] |= target[:, 1:] != target[:, :-1]
    boundary_target[:, :, 1:] |= target[:, :, 1:] != target[:, :, :-1]
    boundary = F.binary_cross_entropy_with_logits(output["boundary"].squeeze(1), boundary_target.float())
    return torch.nan_to_num(0.7 * ce + 0.2 * dice + 0.1 * boundary, nan=0.0)


def evaluate(model, samples, device, num_workers: int):
    import torch
    from torch.utils.data import DataLoader
    from experiment.fewShot_SUIM.data import SUIMDataset, seed_worker

    loader = DataLoader(
        SUIMDataset(samples, crop_size=None, training=False), batch_size=1, shuffle=False,
        num_workers=num_workers, pin_memory=device.type == "cuda", worker_init_fn=seed_worker,
        persistent_workers=num_workers > 0,
    )
    confusion = torch.zeros(NUM_CLASSES, NUM_CLASSES, dtype=torch.int64, device=device)
    model.eval()
    with torch.inference_mode():
        for index, batch in enumerate(loader, 1):
            target = batch["mask"].to(device, non_blocking=True)
            logits = model(batch["image"].to(device, non_blocking=True))["logits"]
            if not torch.isfinite(logits).all():
                raise FloatingPointError("Non-finite evaluation logits")
            prediction = logits.argmax(1)
            confusion += torch.bincount(
                (target * NUM_CLASSES + prediction).reshape(-1), minlength=NUM_CLASSES**2
            ).reshape(NUM_CLASSES, NUM_CLASSES)
            if index % 25 == 0 or index == len(loader):
                print(f"evaluation {index}/{len(loader)}", flush=True)
    cm = confusion.double()
    union = cm.sum(0) + cm.sum(1) - cm.diag()
    iou = cm.diag() / union.clamp_min(1)
    active = union > 0
    if not active.all():
        raise ValueError("Official TEST does not contain every SUIM class")
    return {
        "mIoU": iou.mean().item(),
        "foreground_mIoU": iou[1:].mean().item(),
        "pixel_accuracy": (cm.diag().sum() / cm.sum().clamp_min(1)).item(),
        "per_class_iou": {CLASS_NAMES[c]: iou[c].item() for c in range(NUM_CLASSES)},
        "confusion": confusion.cpu().tolist(),
    }


def main():
    args = parse_args()
    manifests, evaluation, protocol_hash = load_protocol(args.manifest_dir)
    support_keys = manifests[args.shots]["samples"]
    print(json.dumps({
        "shots_per_class": args.shots, "support_images": len(support_keys),
        "evaluation_images": len(evaluation), "protocol_sha256": protocol_hash,
    }, indent=2), flush=True)
    if args.check_only:
        resolve_protocol_paths(args.data_root, manifests[max(SHOTS)]["samples"] + evaluation)
        print("PROTOCOL_OK: nested supports and fixed official TEST files exist", flush=True)
        return
    support, test = resolve_protocol_samples(args.data_root, manifests[args.shots], evaluation)
    if not args.init_checkpoint.is_file():
        raise FileNotFoundError(args.init_checkpoint)

    import numpy as np
    import torch
    from torch.utils.data import DataLoader, RandomSampler
    from experiment.fewShot_SUIM.data import SUIMDataset, seed_worker

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; run training on the GPU server")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    checkpoint_hash = args.checkpoint_sha256 or file_sha256(args.init_checkpoint)
    checkpoint = torch.load(args.init_checkpoint, map_location="cpu", weights_only=False)
    checkpoint.pop("optimizer", None)
    checkpoint.pop("scheduler", None)
    checkpoint.pop("scaler", None)
    implementation_hash = digest({
        str(path.relative_to(ROOT)): path.read_text(encoding="utf-8")
        for path in (
            Path(__file__), Path(__file__).with_name("data.py"),
            Path(__file__).with_name("protocol.py"), Path(__file__).with_name("constants.py"),
            ROOT / "tools/model_1.py",
        )
    })
    settings = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
        if key not in ("output_dir", "skip_completed", "check_only", "checkpoint_sha256")
    }
    signature = digest({
        "protocol": protocol_hash, "checkpoint": checkpoint_hash,
        "implementation": implementation_hash, "settings": settings,
    })
    output = args.output_dir.resolve()
    if output.exists():
        summary_path = output / "summary.json"
        if args.skip_completed and summary_path.is_file():
            result = json.loads(summary_path.read_text(encoding="utf-8"))
            if result.get("signature") != signature:
                raise ValueError(f"Completed run signature changed: {output}")
            print(json.dumps(result, indent=2), flush=True)
            return
        raise FileExistsError(f"Refusing to overwrite {output}")
    device = torch.device(args.device)
    amp = args.amp and device.type == "cuda"
    amp_dtype = torch.bfloat16 if amp and torch.cuda.is_bf16_supported() else torch.float16
    model, transfer = load_v2(checkpoint, args.backbone_path, args.head_init)
    configure_model(model, args.mode)
    model.to(device)
    class_weights, histogram = compute_class_weights(support)
    class_weights = class_weights.to(device)
    output.mkdir(parents=True)
    info = {
        "signature": signature, "protocol_sha256": protocol_hash, "checkpoint_sha256": checkpoint_hash,
        "shots_per_class": args.shots, "support_images": len(support), "evaluation_images": len(test),
        "source_epoch": checkpoint.get("epoch"), "source_val_miou": checkpoint.get("val_miou"),
        "transfer": transfer, "class_histogram": dict(zip(CLASS_NAMES, histogram)),
        "class_weights": dict(zip(CLASS_NAMES, class_weights.cpu().tolist())),
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "total_parameters": sum(p.numel() for p in model.parameters()),
        "trainable_parameter_names": [name for name, p in model.named_parameters() if p.requires_grad],
        "batchnorm": "frozen running statistics", "checkpoint_selection": "fixed final step",
        "evaluation": "official TEST at native resolution, batch size 1, all classes 0..7",
        "precision": str(amp_dtype) if amp else "float32", "config": settings,
    }
    write_json(output / "run_config.json", info)
    write_json(output / "selection.json", manifests[args.shots])

    dataset = SUIMDataset(support, crop_size=args.crop_size, training=True)
    sampler = RandomSampler(
        dataset, replacement=True, num_samples=args.steps * args.batch_size,
        generator=torch.Generator().manual_seed(args.seed),
    )
    loader = DataLoader(
        dataset, batch_size=args.batch_size, sampler=sampler,
        num_workers=args.num_workers, pin_memory=device.type == "cuda", worker_init_fn=seed_worker,
        persistent_workers=args.num_workers > 0, generator=torch.Generator().manual_seed(args.seed),
    )
    encoder = [p for p in model.encoder.parameters() if p.requires_grad]
    encoder_ids = {id(p) for p in encoder}
    other = [p for p in model.parameters() if p.requires_grad and id(p) not in encoder_ids]
    groups = [{"params": other, "lr": args.lr}]
    if encoder:
        groups.append({"params": encoder, "lr": args.lr * args.encoder_lr_multiplier})
    optimizer = torch.optim.AdamW(groups, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: max(0.0, 1.0 - step / args.steps) ** 0.9
    )
    scaler = torch.amp.GradScaler("cuda", enabled=amp and amp_dtype == torch.float16)
    training_mode(model)
    with (output / "history.jsonl").open("w", encoding="utf-8") as history:
        for step, batch in enumerate(loader, 1):
            image = batch["image"].to(device, non_blocking=True)
            target = batch["mask"].to(device, non_blocking=True)
            for _ in range(16):
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast(device.type, enabled=amp, dtype=amp_dtype):
                    prediction = model(image)
                    if not all(torch.isfinite(prediction[key]).all() for key in ("logits", "boundary")):
                        raise FloatingPointError(f"Non-finite output at step {step}")
                    loss = suim_loss(
                        {key: prediction[key].float() for key in ("logits", "boundary")},
                        target, class_weights,
                    )
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Non-finite loss at step {step}")
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                norm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
                if torch.isfinite(norm):
                    scaler.step(optimizer)
                    scaler.update()
                    break
                if not scaler.is_enabled():
                    raise FloatingPointError(f"Non-finite gradients at step {step}")
                scaler.update(new_scale=scaler.get_scale() / 2)
            else:
                raise FloatingPointError(f"Persistent FP16 overflow at step {step}")
            scheduler.step()
            row = {"step": step, "loss": loss.item(), "lr": optimizer.param_groups[0]["lr"]}
            history.write(json.dumps(row) + "\n")
            if step % 20 == 0 or step == args.steps:
                history.flush()
                print(f"step {step}/{args.steps}: loss={loss.item():.4f}", flush=True)
    torch.save({
        "model": model.state_dict(), "model_config": model.model_config, "step": args.steps,
        "experiment": info,
    }, output / "final.pt")
    metrics = evaluate(model, test, device, args.num_workers)
    result = {
        "signature": signature, "mode": args.mode, "steps": args.steps, "shots_per_class": args.shots,
        "seed": args.seed, "support_images": len(support), "evaluation_images": len(test), "metrics": metrics,
    }
    write_json(output / "summary.json", result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
