"""Scan M1's prior modulation strength on a labelled split without training."""

import argparse
import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from experiment.mathseg_uav.model import MathSegUAV
from experiment.mathseg_uav.protocol import (
    boundary_map, confusion_metrics, make_loader, read_splits,
    resolve_dirs, write_json,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=ROOT / "dataset")
    parser.add_argument("--split-dir", type=Path, default=ROOT / "runs/splits")
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-samples", type=int, default=0,
                        help="optional prefix of the split for a quick smoke test")
    parser.add_argument("--factors", nargs="+", type=float,
                        default=[0.0, 0.1, 0.3, 0.5, 1.0, 1.5],
                        help="multipliers in F*(1 + factor*tanh(Conv(prior)))")
    parser.add_argument("--reference-factor", type=float, default=0.1,
                        help="factor used for prediction-change comparisons")
    parser.add_argument("--output", type=Path,
                        help="JSON output; defaults beside the checkpoint")
    args = parser.parse_args()
    if args.num_workers < 0 or args.max_samples < 0:
        parser.error("num-workers and max-samples must be nonnegative")
    values = [*args.factors, args.reference_factor]
    if not values or any(not math.isfinite(value) or value < 0 for value in values):
        parser.error("factors must be finite and nonnegative")
    if len({round(value, 12) for value in args.factors}) != len(args.factors):
        parser.error("factors must be unique")
    return args


def evaluate_factors(model, loader, device, factors, reference_factor):
    all_factors = [reference_factor] + [value for value in factors
                                        if not math.isclose(value, reference_factor, rel_tol=0, abs_tol=1e-12)]
    confusion = {value: torch.zeros(9, 9, dtype=torch.int64, device=device)
                 for value in all_factors}
    boundary_counts = {value: torch.zeros(3, dtype=torch.int64, device=device)
                       for value in all_factors}
    changed = {value: torch.zeros((), dtype=torch.int64, device=device)
               for value in all_factors}
    total_valid = torch.zeros((), dtype=torch.int64, device=device)
    stats = {value: [] for value in all_factors}
    started = time.monotonic()

    with torch.inference_mode():
        for batch in loader:
            image = batch["image"].to(device, non_blocking=True)
            target = batch["mask"].to(device, non_blocking=True)
            valid = target != 0
            features, prior = model.encode_features(image)
            raw = model.head.modulation(prior.to(features[0].dtype))
            reference_prediction = None
            for factor in all_factors:
                delta = factor * torch.tanh(raw)
                output = model.head(features, prior, image.shape[-2:],
                                    modulation_scale=factor)
                logits = output["logits"]
                if not torch.isfinite(logits).all():
                    raise FloatingPointError(f"Non-finite logits at factor {factor}")
                prediction = logits.argmax(1)
                encoded = target[valid] * 9 + prediction[valid]
                confusion[factor] += torch.bincount(encoded, minlength=81).reshape(9, 9)
                actual, _ = boundary_map(target, valid)
                predicted, _ = boundary_map(prediction, valid)
                boundary_counts[factor] += torch.stack(((actual & predicted).sum(),
                                                          predicted.sum(), actual.sum()))
                if math.isclose(factor, reference_factor, rel_tol=0, abs_tol=1e-12):
                    reference_prediction = prediction
                else:
                    if reference_prediction is None:
                        raise RuntimeError("Reference factor must be evaluated first")
                    changed[factor] += ((prediction != reference_prediction) & valid).sum()
                stats[factor].append(torch.stack((delta.abs().mean(), delta.abs().max(),
                                                   delta.std((2, 3)).mean())).cpu())
            total_valid += valid.sum()

    results = []
    valid_pixels = int(total_valid.item())
    for factor in factors:
        metrics = confusion_metrics(confusion[factor])
        tp, predicted, actual = boundary_counts[factor].tolist()
        metrics["boundary_f1_exact"] = 2 * tp / (predicted + actual) if predicted + actual else None
        factor_stats = torch.stack(stats[factor]).mean(0).tolist()
        results.append({
            "factor": factor,
            **metrics,
            "prediction_changed_fraction_vs_reference":
                float(changed[factor].item() / valid_pixels) if factor != reference_factor else 0.0,
            "mean_abs_delta": factor_stats[0],
            "mean_max_abs_delta": factor_stats[1],
            "mean_spatial_std_delta": factor_stats[2],
        })
    return {
        "factors": results,
        "reference_factor": reference_factor,
        "valid_pixels": valid_pixels,
        "samples": len(loader.dataset),
        "seconds": time.monotonic() - started,
    }


def main():
    args = parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; use --device cpu for a smoke test")
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if state.get("experiment") != "mathseg_uav_v1":
        raise ValueError("Expected a MathSeg checkpoint")
    if state["model_config"]["variant"] != "M1":
        raise ValueError("This diagnostic targets an M1 checkpoint")
    images, masks = resolve_dirs(args.data_root)
    splits, hashes = read_splits(args.split_dir, images, masks)
    ids = splits[args.split]
    if args.max_samples:
        ids = ids[:args.max_samples]
    model = MathSegUAV(**state["model_config"]).to(device).eval()
    model.load_state_dict(state["model"], strict=True)
    loader = make_loader(images, masks, ids, workers=args.num_workers)
    result = evaluate_factors(model, loader, device, args.factors, args.reference_factor)
    result.update({
        "checkpoint": str(args.checkpoint),
        "checkpoint_update": state["update"],
        "variant": "M1",
        "split": args.split,
        "split_hash": hashes[args.split],
        "max_samples": args.max_samples,
        "inference": "native_whole_fp32_no_tta",
        "device": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
    })
    output = args.output or args.checkpoint.with_name("modulation_diagnostic.json")
    write_json(output, result)
    print(json.dumps(result, indent=2), flush=True)
    print(f"saved: {output}", flush=True)


if __name__ == "__main__":
    main()
