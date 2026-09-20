"""Native-resolution export and end-to-end model latency measurement."""

import argparse
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from PIL import Image
import torch

from experiment.mathseg_uav.model import MathSegUAV
from experiment.mathseg_uav.protocol import write_json
from experiment.mathseg_uav.train import load_state


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("export", "benchmark"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--size", type=int, default=1024, help="benchmark side length")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--profile-flops", action="store_true", help="partial PyTorch profiler estimate, not total FLOPs")
    parser.add_argument("--allow-any-size", action="store_true", help="export noncompetition images for smoke checks")
    args = parser.parse_args()
    if min(args.size, args.repeats) < 1 or args.warmup < 0:
        parser.error("Invalid benchmark dimensions/counts")
    if args.action == "export" and (args.input is None or args.output is None):
        parser.error("export requires --input and --output")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    state = load_state(args.checkpoint)
    model = MathSegUAV(**state["model_config"]).to(device).eval()
    model.load_state_dict(state["model"], strict=True)
    if args.action == "export":
        paths = sorted(args.input.glob("*.png"))
        if not paths:
            raise FileNotFoundError(f"No PNG images in {args.input}")
        if args.output.resolve() == args.input.resolve():
            raise ValueError("Output must differ from input")
        if args.output.exists() and any(args.output.iterdir()):
            raise FileExistsError("Export requires an empty output directory")
        # Check every input header before writing any predictions.
        if not args.allow_any_size:
            for path in paths:
                with Image.open(path) as image:
                    if image.size != (1024, 1024):
                        raise ValueError(f"Expected 1024x1024: {path}")
        args.output.mkdir(parents=True, exist_ok=True)
        for index, path in enumerate(paths, 1):
            with Image.open(path) as image:
                array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255
            tensor = torch.from_numpy(array).permute(2, 0, 1)[None].to(device)
            logits = model(tensor)["logits"]
            if not torch.isfinite(logits).all():
                raise FloatingPointError(f"Non-finite logits: {path}")
            prediction = logits.argmax(1)[0].cpu().numpy().astype(np.uint8)
            Image.fromarray(prediction).save(args.output / path.name)
            if index % 50 == 0 or index == len(paths):
                print(f"exported {index}/{len(paths)}", flush=True)
        write_json(args.output.parent / (args.output.name + "_manifest.json"), {
            "checkpoint": str(args.checkpoint), "variant": model.variant, "count": len(paths),
            "mode": "native_whole_fp32_no_tta", "filenames": [p.name for p in paths],
        })
        return
    image = torch.rand(1, 3, args.size, args.size, device=device)
    sync = (lambda: torch.cuda.synchronize(device)) if device.type == "cuda" else (lambda: None)
    for _ in range(args.warmup):
        model(image)
    sync()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    durations = []
    for _ in range(args.repeats):
        sync()
        start = time.perf_counter()
        model(image)
        sync()
        durations.append((time.perf_counter() - start) * 1000)
    result = {
        "variant": model.variant, "checkpoint": str(args.checkpoint), "size": args.size,
        "batch_size": 1, "precision": "fp32", "warmup": args.warmup, "repeats": args.repeats,
        "latency_mean_ms": statistics.mean(durations), "latency_median_ms": statistics.median(durations),
        "latency_p95_ms": float(np.percentile(durations, 95)),
        "images_per_second": 1000 / statistics.mean(durations),
        "parameters": sum(p.numel() for p in model.parameters()),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
        "device": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "torch_version": torch.__version__, "cpu_threads": torch.get_num_threads(),
        "scope": "full forward including priors, heads and resize; excludes image I/O and CPU/GPU transfer",
    }
    if args.profile_flops:
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU], with_flops=True) as profile:
            model(image)
            sync()
        result["profiler_partial_flops"] = sum(event.flops for event in profile.key_averages())
        result["flops_warning"] = "Partial estimate: unsupported operators omitted; not total model FLOPs."
    output = args.output or args.checkpoint.parent / "benchmark_fp32.json"
    write_json(output, result)
    print(result)


if __name__ == "__main__":
    main()
