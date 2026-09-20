"""Run fixed-holdout 0/1/2/5/10-shot experiments using HyperSeg-UAV v2."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiment.loveda_fewshot.manual_protocol import SHOTS, digest, load_protocol, resolve_samples


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def load_v2(checkpoint, backbone_path=None):
    from tools.model_1 import HyperSegUAV
    from experiment.loveda_fewshot.train import legacy_encoder_key

    config = dict(checkpoint["model_config"])
    if config.pop("version", None) != "v2" or config.get("classes") != 9:
        raise ValueError("Expected an original 9-class UAV v2 checkpoint")
    config.update(classes=8, pretrained=False)
    local = ROOT / "models" / config.get("model_name", "nvidia/mit-b3").replace("/", "--")
    if backbone_path or local.is_dir():
        config["model_name"] = str(backbone_path or local)
    model = HyperSegUAV(**config)
    destination = model.state_dict()
    mapped = {}
    for key, tensor in checkpoint["model"].items():
        target = key if key in destination else legacy_encoder_key(key)
        if target is None or target not in destination:
            raise ValueError(f"Unexpected checkpoint tensor: {key}")
        if target in mapped:
            raise ValueError(f"Multiple source tensors map to {target}")
        # Vegetation -> Forest is an approximate semantic mapping; Vehicle is dropped.
        if target in ("head.weight", "head.bias"):
            if tensor.shape[0] != 9:
                raise ValueError("Expected 9-channel UAV head")
            tensor = tensor[:8].clone()
        if tensor.shape != destination[target].shape:
            raise ValueError(f"Shape mismatch for {target}: {tensor.shape} vs {destination[target].shape}")
        mapped[target] = tensor
    model.load_state_dict(mapped, strict=True)
    return model


def configure_model(model, mode):
    if mode == "semantic-head":
        for parameter in model.parameters():
            parameter.requires_grad = False
        for parameter in model.head.parameters():
            parameter.requires_grad = True
        return
    from experiment.loveda_fewshot.train import configure_trainable
    configure_trainable(model, mode)
    if mode == "adapter":
        for parameter in model.boundary_refine.parameters():
            parameter.requires_grad = True


def training_mode(model):
    from torch import nn
    from experiment.loveda_fewshot.train import keep_frozen_modules_in_eval
    model.train()
    keep_frozen_modules_in_eval(model)
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.eval()


def run_one(args, manifest, evaluation, paths, checkpoint, k, seed, mode, output, signature):
    import numpy as np
    import torch
    from torch.utils.data import DataLoader, RandomSampler
    from experiment.loveda_fewshot.train import (
        Sample, LoveDADataset, seed_worker, compute_class_weights, loveda_loss, CLASS_NAMES,
    )
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = torch.device(args.device)
    amp = args.amp and device.type == "cuda"
    amp_dtype = torch.bfloat16 if amp and torch.cuda.is_bf16_supported() else torch.float16
    model = load_v2(checkpoint, args.backbone_path)
    configure_model(model, mode)
    model.to(device)
    def samples(keys):
        return [Sample(key.split("/")[0], key.split("/")[1], *paths[key]) for key in keys]
    common = dict(num_workers=args.num_workers, pin_memory=device.type == "cuda", worker_init_fn=seed_worker)
    info = dict(signature=signature, shots_per_class=k, seed=seed if k else None,
                mode=mode if k else "zero", support_images=len(manifest["samples"]),
                evaluation_images=len(evaluation), model_version="v2",
                source_epoch=checkpoint.get("epoch"), source_val_miou=checkpoint.get("val_miou"),
                head_mapping={str(c): c for c in range(8)},
                forest_mapping="source Vegetation -> target Forest (approximate)",
                checkpoint_selection="fixed final step; no evaluation-based selection",
                batchnorm="frozen running statistics",
                trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                total_parameters=sum(p.numel() for p in model.parameters()),
                trainable_parameter_names=[name for name, p in model.named_parameters() if p.requires_grad],
                precision=str(amp_dtype) if amp else "float32", config=vars(args).copy())
    info["config"] = {key: str(value) if isinstance(value, Path) else value for key, value in info["config"].items()}
    write_json(output / "run_config.json", info)
    write_json(output / "selection.json", manifest)
    if k:
        support = samples(manifest["samples"])
        weights, histogram = compute_class_weights(support)
        weights = weights.to(device)
        info["class_histogram"] = dict(zip(CLASS_NAMES, histogram))
        info["class_weights"] = weights.tolist()
        write_json(output / "run_config.json", info)
        dataset = LoveDADataset(support, args.crop_size, training=True)
        generator = torch.Generator().manual_seed(seed)
        sampler = RandomSampler(dataset, replacement=True, num_samples=args.steps * args.batch_size, generator=generator)
        loader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler,
                            generator=torch.Generator().manual_seed(seed), **common)
        encoder = [p for p in model.encoder.parameters() if p.requires_grad]
        encoder_ids = {id(p) for p in encoder}
        other = [p for p in model.parameters() if p.requires_grad and id(p) not in encoder_ids]
        groups = [{"params": other, "lr": args.lr}]
        if encoder:
            groups.append({"params": encoder, "lr": args.lr * 0.1})
        optimizer = torch.optim.AdamW(groups, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: max(0., 1 - step / args.steps) ** .9)
        scaler = torch.amp.GradScaler("cuda", enabled=amp and amp_dtype == torch.float16)
        training_mode(model)
        with (output / "history.jsonl").open("w", encoding="utf-8") as history:
            for step, batch in enumerate(loader, 1):
                image, target = batch["image"].to(device), batch["mask"].to(device)
                for attempt in range(16):
                    optimizer.zero_grad(set_to_none=True)
                    with torch.amp.autocast("cuda", enabled=amp, dtype=amp_dtype):
                        prediction = model(image)
                        if not all(torch.isfinite(prediction[key]).all() for key in ("logits", "boundary")):
                            raise FloatingPointError(f"Non-finite output at step {step}")
                        loss = loveda_loss({key: prediction[key].float() for key in ("logits", "boundary")}, target, weights)
                    if not torch.isfinite(loss):
                        raise FloatingPointError(f"Non-finite loss at step {step}")
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    norm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
                    if not torch.isfinite(norm):
                        if not scaler.is_enabled():
                            raise FloatingPointError(f"Non-finite gradients at step {step}")
                        # Retry this batch with a lower scale; do not count a skipped update.
                        scaler.update(new_scale=scaler.get_scale() / 2)
                        print(f"step {step}: retry FP16 overflow, scale={scaler.get_scale()}", flush=True)
                        continue
                    scaler.step(optimizer)
                    scaler.update()
                    break
                else:
                    raise FloatingPointError(f"Persistent FP16 overflow at step {step}")
                scheduler.step()
                row = dict(step=step, loss=loss.item(), lr=optimizer.param_groups[0]["lr"])
                history.write(json.dumps(row) + "\n")
                if step % 20 == 0 or step == args.steps:
                    history.flush()
                    print(f"{output.name} step {step}/{args.steps}: loss={loss.item():.4f}", flush=True)
        torch.save(dict(model=model.state_dict(), model_config=model.model_config, step=args.steps,
                        experiment=info), output / "final.pt")
        del optimizer, loader, scaler

    loader = DataLoader(LoveDADataset(samples(evaluation), args.eval_size, training=False),
                        batch_size=args.eval_batch_size, shuffle=False, **common)
    model.eval()
    confusion = torch.zeros(8, 8, dtype=torch.int64, device=device)
    with torch.inference_mode():
        for index, batch in enumerate(loader, 1):
            target = batch["mask"].to(device)
            logits = model(batch["image"].to(device))["logits"]
            if not torch.isfinite(logits).all():
                raise FloatingPointError("Non-finite evaluation logits")
            prediction = logits[:, 1:].argmax(1) + 1
            valid = target != 0
            confusion += torch.bincount(target[valid] * 8 + prediction[valid], minlength=64).reshape(8, 8)
            if index % 100 == 0:
                print(f"{output.name} evaluation {index}/{len(loader)}", flush=True)
    cm = confusion.double()
    union = cm.sum(0) + cm.sum(1) - cm.diag()
    iou = cm.diag() / union.clamp_min(1)
    active = union[1:] > 0
    if not active.any():
        raise ValueError("No valid evaluation pixels")
    metrics = dict(mIoU=iou[1:][active].mean().item(),
                   per_class_iou={CLASS_NAMES[c]: iou[c].item() if union[c] > 0 else None for c in range(1, 8)},
                   pixel_accuracy=(cm.diag().sum() / cm.sum().clamp_min(1)).item(),
                   confusion=confusion.cpu().tolist())
    result = dict(signature=signature, mode=mode if k else "zero", shots_per_class=k,
                  seed=seed if k else None, support_images=len(manifest["samples"]),
                  evaluation_images=len(evaluation), metrics=metrics)
    write_json(output / "summary.json", result)
    print(json.dumps(result), flush=True)
    return result


def summarize(output, results):
    rows = [dict(mode=r["mode"], shots_per_class=r["shots_per_class"], seed=r["seed"],
                 support_images=r["support_images"], mIoU=r["metrics"]["mIoU"], **r["metrics"]["per_class_iou"])
            for r in results]
    with (output / "results.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    aggregates = []
    for mode, k in sorted({(r["mode"], r["shots_per_class"]) for r in results}):
        values = [r["metrics"]["mIoU"] for r in results if (r["mode"], r["shots_per_class"]) == (mode, k)]
        aggregates.append(dict(mode=mode, shots_per_class=k, runs=len(values), mean_mIoU=statistics.mean(values),
                               std_mIoU=statistics.stdev(values) if len(values) > 1 else 0.0))
    write_json(output / "aggregate.json", {"variation": "training seeds; support selection is fixed", "results": aggregates})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=ROOT / "LoveDA")
    parser.add_argument("--manifest-dir", type=Path, required=True)
    parser.add_argument("--init-checkpoint", type=Path, default=ROOT / "models/hyperseg_resume_best.pt")
    parser.add_argument("--backbone-path", type=Path, help="Local Hugging Face mit-b3 directory with config and weights")
    parser.add_argument("--output-root", type=Path, default=ROOT / "runs/loveda_manual_v2")
    parser.add_argument("--shots", nargs="+", type=int, choices=SHOTS, default=list(SHOTS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[3407, 3408, 3409])
    parser.add_argument("--modes", nargs="+", choices=("head", "semantic-head", "adapter", "full"), default=["adapter"])
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--crop-size", type=int, default=512)
    parser.add_argument("--eval-size", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--eval-batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--check-only", action="store_true", help="Validate manifests and files without PyTorch or training")
    parser.add_argument("--skip-completed", action="store_true", help="Reuse completed runs only if signatures match")
    args = parser.parse_args()
    if min(args.steps, args.crop_size, args.eval_size, args.batch_size, args.eval_batch_size) < 1 or args.lr <= 0 or args.num_workers < 0:
        parser.error("Sizes, steps and learning rate must be positive; workers cannot be negative")
    for name in ("shots", "seeds", "modes"):
        if len(getattr(args, name)) != len(set(getattr(args, name))):
            parser.error(f"Duplicate --{name}")
    manifests, evaluation, protocol_hash = load_protocol(args.manifest_dir)
    paths = resolve_samples(args.data_root, manifests[10]["samples"] + evaluation)
    if not args.init_checkpoint.is_file():
        raise FileNotFoundError(args.init_checkpoint)
    print(json.dumps(dict(support_images={k: len(m["samples"]) for k, m in manifests.items()},
                          evaluation_images=len(evaluation), protocol_sha256=protocol_hash), indent=2), flush=True)
    if args.check_only:
        print("PROTOCOL_OK: fixed evaluation, disjoint supports, nested per-class prefixes, image/mask files exist")
        return
    import torch
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Run on the GPU server; use --check-only for local checks.")
    sha = hashlib.sha256()
    with args.init_checkpoint.open("rb") as file:
        for block in iter(lambda: file.read(8 * 1024 * 1024), b""):
            sha.update(block)
    checkpoint = torch.load(args.init_checkpoint, map_location="cpu", weights_only=False)
    checkpoint.pop("optimizer", None)
    checkpoint.pop("scheduler", None)
    checkpoint.pop("scaler", None)
    args.checkpoint_sha256 = sha.hexdigest()
    args.protocol_sha256 = protocol_hash
    implementation_hash = digest({str(path.relative_to(ROOT)): path.read_text(encoding="utf-8") for path in (
        Path(__file__), Path(__file__).with_name("manual_protocol.py"), Path(__file__).with_name("train.py"),
        ROOT / "tools/model_1.py",
    )})
    args.output_root.mkdir(parents=True, exist_ok=True)
    results = []
    settings = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()
                if key not in ("output_root", "skip_completed", "shots", "seeds", "modes", "check_only")}
    jobs = [(0, 3407, "zero")] if 0 in args.shots else []
    jobs += [(k, seed, mode) for k in args.shots if k for mode in args.modes for seed in args.seeds]
    for k, seed, mode in jobs:
        name = "zero_0shot" if not k else f"{mode}_{k}shot_seed{seed}"
        output = args.output_root / name
        signature = digest(dict(protocol=protocol_hash, checkpoint=sha.hexdigest(), settings=settings,
                                k=k, seed=seed, mode=mode, implementation=implementation_hash))
        if output.exists():
            summary = output / "summary.json"
            if args.skip_completed and summary.is_file():
                result = json.loads(summary.read_text())
                if result.get("signature") != signature:
                    raise ValueError(f"Run configuration changed: {output}; use a new output root")
            else:
                raise FileExistsError(f"Refusing to overwrite {output}; use a new output root or --skip-completed for completed runs")
        else:
            output.mkdir()
            result = run_one(args, manifests[k], evaluation, paths, checkpoint, k, seed,
                             "head" if not k else mode, output, signature)
        results.append(result)
        summarize(args.output_root, results)


if __name__ == "__main__":
    main()
