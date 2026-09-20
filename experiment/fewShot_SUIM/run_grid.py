"""Run the SUIM 200/2000-step adapter/classification-head experiment grid."""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import subprocess
import sys
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RUN_SCRIPT = Path(__file__).with_name("run.py")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SETTINGS = {
    "adapter_200": ("adapter", 200),
    "adapter_2000": ("adapter", 2000),
    "semantic_head_200": ("semantic-head", 200),
    "semantic_head_2000": ("semantic-head", 2000),
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=ROOT / "SUIM")
    parser.add_argument("--manifest-dir", type=Path, required=True)
    parser.add_argument("--init-checkpoint", type=Path, default=ROOT / "models/hyperseg_resume_best.pt")
    parser.add_argument("--backbone-path", type=Path)
    parser.add_argument("--output-root", type=Path, default=ROOT / "runs/suim_fewshot_v1")
    parser.add_argument("--settings", nargs="+", choices=tuple(SETTINGS), default=list(SETTINGS))
    parser.add_argument("--shots", nargs="+", type=int, choices=(1, 2, 5, 10), default=[1, 2, 5, 10])
    parser.add_argument("--seeds", nargs="+", type=int, default=[3407, 3408, 3409])
    parser.add_argument("--head-init", choices=("random", "semantic-map"), default="random")
    parser.add_argument("--crop-size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-completed", action="store_true")
    args = parser.parse_args()
    for field in ("settings", "shots", "seeds"):
        values = getattr(args, field)
        if len(values) != len(set(values)):
            parser.error(f"Duplicate --{field} values")
    if min(args.crop_size, args.batch_size) < 1 or args.num_workers < 0 or args.lr <= 0:
        parser.error("Invalid loader or optimizer settings")
    return args


def commands(args, checkpoint_hash=None):
    for setting in args.settings:
        mode, steps = SETTINGS[setting]
        for shots in args.shots:
            for seed in args.seeds:
                output = args.output_root / setting / f"{shots}shot_seed{seed}"
                command = [
                    sys.executable, "-u", str(RUN_SCRIPT),
                    "--data-root", str(args.data_root),
                    "--manifest-dir", str(args.manifest_dir),
                    "--init-checkpoint", str(args.init_checkpoint),
                    "--output-dir", str(output),
                    "--shots", str(shots), "--seed", str(seed),
                    "--mode", mode, "--steps", str(steps),
                    "--head-init", args.head_init,
                    "--crop-size", str(args.crop_size), "--batch-size", str(args.batch_size),
                    "--num-workers", str(args.num_workers), "--lr", str(args.lr),
                    "--device", args.device, "--amp" if args.amp else "--no-amp",
                ]
                if checkpoint_hash:
                    command += ["--checkpoint-sha256", checkpoint_hash]
                if args.backbone_path:
                    command += ["--backbone-path", str(args.backbone_path)]
                if args.skip_completed:
                    command.append("--skip-completed")
                yield setting, mode, steps, shots, seed, output, command


def summarize(args):
    rows = []
    for setting, mode, steps, shots, seed, output, _ in commands(args):
        summary_path = output / "summary.json"
        if not summary_path.is_file():
            continue
        result = json.loads(summary_path.read_text(encoding="utf-8"))
        metrics = result["metrics"]
        row = {
            "setting": setting, "mode": mode, "steps": steps, "shots_per_class": shots,
            "seed": seed, "support_images": result["support_images"],
            "mIoU": metrics["mIoU"], "foreground_mIoU": metrics["foreground_mIoU"],
            "pixel_accuracy": metrics["pixel_accuracy"],
        }
        row.update(metrics["per_class_iou"])
        rows.append(row)
    if not rows:
        return
    args.output_root.mkdir(parents=True, exist_ok=True)
    with (args.output_root / "results.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    aggregates = []
    for setting in args.settings:
        mode, steps = SETTINGS[setting]
        for shots in args.shots:
            selected = [row for row in rows if row["setting"] == setting and row["shots_per_class"] == shots]
            if not selected:
                continue
            values = [row["mIoU"] for row in selected]
            aggregates.append({
                "setting": setting, "mode": mode, "steps": steps, "shots_per_class": shots,
                "runs": len(values), "mean_mIoU": statistics.mean(values),
                "std_mIoU": statistics.stdev(values) if len(values) > 1 else 0.0,
            })
    (args.output_root / "aggregate.json").write_text(
        json.dumps({"results": aggregates}, indent=2) + "\n", encoding="utf-8"
    )


def main():
    args = parse_args()
    jobs = list(commands(args))
    if args.dry_run:
        for setting, _, _, shots, seed, _, command in jobs:
            print(f"{setting} K={shots} seed={seed}: {subprocess.list2cmdline(command)}")
        print(f"{len(jobs)} runs total")
        return
    if not args.init_checkpoint.is_file():
        raise FileNotFoundError(args.init_checkpoint)
    from experiment.fewShot_SUIM.run import file_sha256

    checkpoint_hash = file_sha256(args.init_checkpoint)
    jobs = list(commands(args, checkpoint_hash))
    args.output_root.mkdir(parents=True, exist_ok=True)
    log_path = args.output_root / ("grid_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f") + ".log")
    with log_path.open("x", encoding="utf-8") as log:
        def emit(line):
            print(line, end="", flush=True)
            log.write(line)
            log.flush()

        emit(f"Unified stdout/stderr log: {log_path}\n")
        for setting, _, _, shots, seed, _, command in jobs:
            emit(f"\nSTART {setting} K={shots} seed={seed}\n")
            with subprocess.Popen(
                command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, encoding="utf-8", errors="replace", bufsize=1,
            ) as process:
                for line in process.stdout:
                    emit(line)
                status = process.wait()
            summarize(args)
            if status:
                emit(f"FAILED {setting} K={shots} seed={seed}: exit={status}\n")
                raise SystemExit(status)
            emit(f"DONE {setting} K={shots} seed={seed}\n")
        emit("ALL_DONE\n")


if __name__ == "__main__":
    main()
