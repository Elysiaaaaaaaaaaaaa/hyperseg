"""Single-seed update-scope/budget ablation: three new settings, each at K=1,2."""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SETTINGS = (("adapter_200", "adapter", 200),
            ("semantic_head_2000", "semantic-head", 2000),
            ("semantic_head_200", "semantic-head", 200))


def commands(args):
    for name, mode, steps in SETTINGS:
        command = [sys.executable, "-u", str(Path(__file__).with_name("run_manual.py")),
                   "--data-root", str(args.data_root), "--manifest-dir", str(args.manifest_dir),
                   "--init-checkpoint", str(args.init_checkpoint),
                   "--output-root", str(args.output_root / name),
                   "--shots", "1", "2", "--seeds", str(args.seed), "--modes", mode,
                   "--steps", str(steps), "--batch-size", "2", "--crop-size", "512",
                   "--eval-size", "1024", "--eval-batch-size", "1", "--lr", "0.0001",
                   "--num-workers", str(args.num_workers), "--amp", "--device", "cuda"]
        if args.backbone_path:
            command += ["--backbone-path", str(args.backbone_path)]
        if args.skip_completed:
            command.append("--skip-completed")
        yield name, mode, steps, command


def summarize(args):
    rows = []
    for name, mode, steps in SETTINGS:
        for k in (1, 2):
            folder = args.output_root / name / f"{mode}_{k}shot_seed{args.seed}"
            if not (folder / "summary.json").is_file():
                continue
            result = json.loads((folder / "summary.json").read_text())
            config = json.loads((folder / "run_config.json").read_text())
            rows.append(dict(setting=name, shots_per_class=k, seed=args.seed, steps=steps,
                             support_images=result["support_images"],
                             trainable_parameters=config["trainable_parameters"],
                             mIoU=result["metrics"]["mIoU"], **result["metrics"]["per_class_iou"]))
    if rows:
        with (args.output_root / "ablation_results.csv").open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=ROOT / "LoveDA")
    parser.add_argument("--manifest-dir", type=Path, required=True)
    parser.add_argument("--init-checkpoint", type=Path, default=ROOT / "models/hyperseg_resume_best.pt")
    parser.add_argument("--output-root", type=Path, default=ROOT / "runs/loveda_ablation_v2")
    parser.add_argument("--backbone-path", type=Path)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--dry-run", action="store_true", help="Print exact three settings without running or writing files")
    parser.add_argument("--skip-completed", action="store_true")
    args = parser.parse_args()
    if args.num_workers < 0:
        parser.error("--num-workers must be nonnegative")
    jobs = list(commands(args))
    if args.dry_run:
        for name, _, _, command in jobs:
            print(name + ": " + subprocess.list2cmdline(command))
        print("6 runs total: three settings x K=1,2; one seed; no 0-shot or adapter-2000 baseline.")
        return
    args.output_root.mkdir(parents=True, exist_ok=True)
    log_path = args.output_root / ("ablation_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f") + ".log")
    with log_path.open("x", encoding="utf-8") as log:
        def emit(line):
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
        emit(f"Unified stdout/stderr log: {log_path}\n")
        for name, _, _, command in jobs:
            emit(f"\nSTART {name}: {subprocess.list2cmdline(command)}\n")
            with subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                  text=True, encoding="utf-8", errors="replace", bufsize=1) as process:
                for line in process.stdout:
                    emit(line)
                status = process.wait()
            summarize(args)
            if status:
                emit(f"FAILED {name}: exit={status}; remaining settings were not launched.\n")
                raise SystemExit(status)
            emit(f"DONE {name}\n")
        emit("ALL_DONE: six ablation runs completed.\n")


if __name__ == "__main__":
    main()
