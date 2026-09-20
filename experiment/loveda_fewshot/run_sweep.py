"""Run and summarize a LoveDA few-shot mode/shot/seed sweep."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
TRAIN_SCRIPT = Path(__file__).with_name("train.py")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--init-checkpoint", type=Path, default=ROOT / "models/hyperseg_b3_best.pt")
    parser.add_argument("--output-root", type=Path, default=ROOT / "runs/loveda_fewshot")
    parser.add_argument("--shots", nargs="+", type=int, default=(1, 5, 10, 20))
    parser.add_argument("--seeds", nargs="+", type=int, default=(3407, 3408, 3409))
    parser.add_argument("--modes", nargs="+", choices=("head", "adapter", "full"), default=("head", "adapter", "full"))
    parser.add_argument("extra", nargs=argparse.REMAINDER, help="Arguments passed to train.py after '--'")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    rows = []
    extra = args.extra[1:] if args.extra[:1] == ["--"] else args.extra
    for shots in args.shots:
        for seed in args.seeds:
            seed_manifest = args.output_root / "manifests" / f"{shots}shot_seed{seed}.json"
            for mode_index, mode in enumerate(args.modes):
                output_dir = args.output_root / f"{mode}_{shots}shot_seed{seed}"
                command = [
                    sys.executable,
                    str(TRAIN_SCRIPT),
                    "--data-root", str(args.data_root),
                    "--init-checkpoint", str(args.init_checkpoint),
                    "--output-dir", str(output_dir),
                    "--shots", str(shots),
                    "--seed", str(seed),
                    "--mode", mode,
                ]
                if mode_index > 0:
                    command.extend(("--manifest", str(seed_manifest)))
                command.extend(extra)
                subprocess.run(command, check=True, cwd=ROOT)
                if mode_index == 0:
                    seed_manifest.parent.mkdir(parents=True, exist_ok=True)
                    seed_manifest.write_text(
                        (output_dir / "selection.json").read_text(encoding="utf-8"), encoding="utf-8"
                    )
                summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
                rows.append({
                    "mode": mode,
                    "shots_per_domain": shots,
                    "seed": seed,
                    "best_mIoU": summary["best_mIoU"],
                })

    with (args.output_root / "sweep_results.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=("mode", "shots_per_domain", "seed", "best_mIoU"))
        writer.writeheader()
        writer.writerows(rows)

    aggregates = []
    for shots in args.shots:
        for mode in args.modes:
            values = [row["best_mIoU"] for row in rows if row["shots_per_domain"] == shots and row["mode"] == mode]
            aggregates.append({
                "mode": mode,
                "shots_per_domain": shots,
                "runs": len(values),
                "mean_mIoU": statistics.mean(values),
                "std_mIoU": statistics.stdev(values) if len(values) > 1 else 0.0,
            })
    (args.output_root / "sweep_summary.json").write_text(
        json.dumps(aggregates, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(aggregates, indent=2))


if __name__ == "__main__":
    main()
