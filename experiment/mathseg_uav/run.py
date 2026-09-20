"""Plan/run a sequential ablation queue, or summarize results (standard library only)."""

import argparse
import csv
import json
import os
import shlex
import statistics
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def summarize(root):
    rows, groups = [], {}
    for config_path in sorted(root.glob("*/config.json")):
        directory = config_path.parent
        config = json.loads(config_path.read_text())
        best_path, status_path = directory / "best_metrics.json", directory / "status.json"
        best = json.loads(best_path.read_text()) if best_path.exists() else {}
        status = json.loads(status_path.read_text()) if status_path.exists() else {}
        benchmark_path = directory / "benchmark_fp32.json"
        benchmark = json.loads(benchmark_path.read_text()) if benchmark_path.exists() else {}
        row = {"run": directory.name, "variant": config["contract"]["variant"],
               "seed": config["contract"]["seed"], "completed": status.get("completed", False),
               "best_update": best.get("update"), "val_mIoU": best.get("mIoU"),
               "boundary_f1_exact": best.get("boundary_f1_exact"), "parameters": config["parameters"],
               "latency_mean_ms": benchmark.get("latency_mean_ms"),
               "peak_allocated_bytes": benchmark.get("peak_allocated_bytes"),
               "benchmark_device": benchmark.get("device"), "benchmark_size": benchmark.get("size")}
        for cls in range(1, 9):
            row[f"IoU_{cls}"] = best.get("per_class_iou", {}).get(str(cls))
        rows.append(row)
        # Never aggregate short/incomplete runs, or mix different protocols.
        if row["completed"] and row["val_mIoU"] is not None:
            protocol = {k: v for k, v in config["contract"].items() if k != "seed"}
            protocol["initialization"] = {"model_name": config["args"]["model_name"],
                                           "pretrained": not config["args"]["no_pretrained"],
                                           "encoder_config": config["model_config"]["encoder_config"]}
            key = json.dumps(protocol, sort_keys=True)
            groups.setdefault(key, []).append(row)
    if not rows:
        raise FileNotFoundError(f"No */config.json under {root}")
    with (root / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    aggregates = []
    for key, group in groups.items():
        seeds = [row["seed"] for row in group]
        if len(set(seeds)) != len(seeds):
            raise ValueError("Duplicate seeds within a protocol; refuse to count them as independent runs")
        scores = [row["val_mIoU"] for row in group]
        aggregates.append({"protocol": json.loads(key), "seeds": seeds, "n": len(scores),
                           "mean_mIoU": statistics.mean(scores),
                           "sample_std_mIoU": statistics.stdev(scores) if len(scores) > 1 else None})
    (root / "aggregate.json").write_text(json.dumps(aggregates, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(rows, indent=2))
    print(f"Saved {root / 'summary.csv'} and {root / 'aggregate.json'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("plan", "run", "summarize"))
    parser.add_argument("--variants", nargs="+", choices=("M0", "M1", "M2", "M3", "M4"), default=["M0", "M1", "M2", "M3"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[3407])
    parser.add_argument("--output-root", type=Path, default=ROOT / "runs/mathseg_uav")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--resume", action="store_true", help="resume last.pt; skip completed runs")
    # Train options go after a literal --, preserving their exact argument boundaries.
    argv = sys.argv[1:]
    separator = argv.index("--") if "--" in argv else len(argv)
    args = parser.parse_args(argv[:separator])
    args.output_root = args.output_root.resolve()
    extra = argv[separator + 1:]
    reserved = {"--variant", "--seed", "--work-dir", "--resume", "--eval-checkpoint", "--check-only"}
    if any(value.split("=")[0] in reserved for value in extra):
        parser.error("Queue-owned options cannot be forwarded to train.py")
    if len(set(args.seeds)) != len(args.seeds) or len(set(args.variants)) != len(args.variants):
        parser.error("Duplicate seeds or variants")
    if args.action == "summarize":
        summarize(args.output_root)
        return
    commands = []
    for seed in args.seeds:
        for variant in args.variants:
            directory = args.output_root / f"{variant}_seed{seed}"
            command = [args.python, "-u", str(Path(__file__).with_name("train.py")),
                       "--variant", variant, "--seed", str(seed), "--work-dir", str(directory), *extra]
            if args.resume and (directory / "last.pt").exists():
                command += ["--resume", str(directory / "last.pt")]
            commands.append((directory, command))
    if args.action == "plan":
        for _, command in commands:
            print(shlex.join(command))
        return
    args.output_root.mkdir(parents=True, exist_ok=True)
    lock = args.output_root / "queue.lock"
    descriptor = os.open(str(lock), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    try:
        os.write(descriptor, str(os.getpid()).encode())
        os.close(descriptor)
        for directory, command in commands:
            status = directory / "status.json"
            if args.resume and status.exists() and json.loads(status.read_text()).get("completed"):
                print(f"skip completed: {directory}", flush=True)
                continue
            if not args.resume and any((directory / name).exists() for name in ("config.json", "last.pt", "best.pt")):
                raise FileExistsError(f"{directory} already exists; select --resume or another output root")
            directory.mkdir(parents=True, exist_ok=True)
            print(shlex.join(command), flush=True)
            with (directory / "train.log").open("a", encoding="utf-8") as log:
                result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            (directory / "exit_code").write_text(str(result.returncode) + "\n")
            if result.returncode:
                raise SystemExit(f"Failed ({result.returncode}); inspect {directory / 'train.log'}")
        summarize(args.output_root)
    finally:
        lock.unlink()


if __name__ == "__main__":
    main()
