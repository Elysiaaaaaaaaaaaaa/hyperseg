"""Audit SUIM and create one fixed, nested 1/2/5/10-shot protocol."""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiment.fewShot_SUIM.constants import CLASS_NAMES, NUM_CLASSES
from experiment.fewShot_SUIM.data import discover_samples, read_sample
from experiment.fewShot_SUIM.protocol import SHOTS, class_metadata, load_protocol, write_json


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=ROOT / "SUIM")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "runs/suim_fewshot_protocol_seed3407")
    parser.add_argument("--selection-seed", type=int, default=3407)
    parser.add_argument("--min-class-pixels", type=int, default=512,
                        help="Minimum target pixels required for an image to enter a class pool")
    parser.add_argument("--min-class-percent", type=float, default=0.05,
                        help="Minimum target area percentage required in addition to --min-class-pixels")
    parser.add_argument("--selection-file", type=Path,
                        help="Optional JSON containing ordered per_class lists; replaces random selection")
    args = parser.parse_args()
    if args.min_class_pixels < 1 or not 0 <= args.min_class_percent <= 100:
        parser.error("Selection thresholds are invalid")
    return args


def normalize_manual_selection(path: Path) -> dict[str, list[str]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    per_class = payload.get("per_class", payload)
    if not isinstance(per_class, dict) or set(per_class) != {str(c) for c in range(NUM_CLASSES)}:
        raise ValueError("Manual selection must contain per_class keys 0..7")
    result = {}
    for class_id in range(NUM_CLASSES):
        values = per_class[str(class_id)]
        if not isinstance(values, list) or len(values) != 10 or len(set(values)) != 10:
            raise ValueError(f"Class {class_id} must contain exactly 10 unique ordered images")
        result[str(class_id)] = [value if "/" in value else f"train_val/{value}" for value in values]
    return result


def main():
    args = parse_args()
    data_root, output = args.data_root.resolve(), args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    protected = [output / f"{k}shot.json" for k in SHOTS] + [output / "evaluation.json"]
    if any(path.exists() for path in protected):
        raise FileExistsError(f"Refusing to overwrite an existing protocol in {output}")

    train, test = discover_samples(data_root, "train_val"), discover_samples(data_root, "test")
    records = {}
    total_histogram = np.zeros(NUM_CLASSES, dtype=np.int64)
    image_presence = np.zeros(NUM_CLASSES, dtype=np.int64)
    off_palette_pixels = far_pixels = total_pixels = repaired_masks = off_palette_masks = 0
    for index, sample in enumerate(train, 1):
        _, mask, audit = read_sample(sample)
        counts = np.bincount(mask.reshape(-1), minlength=NUM_CLASSES)[:NUM_CLASSES]
        total_histogram += counts
        image_presence += counts > 0
        total_pixels += int(audit["pixels"])
        off_palette_pixels += int(audit["off_palette_pixels"])
        far_pixels += int(audit["far_from_palette_pixels"])
        repaired_masks += int(audit["size_repaired"])
        off_palette_masks += int(audit["off_palette_pixels"] > 0)
        records[sample.key] = {"counts": counts.tolist(), "pixels": int(mask.size)}
        if index % 100 == 0:
            print(f"audit {index}/{len(train)}", flush=True)

    if args.selection_file:
        per_class = normalize_manual_selection(args.selection_file)
    else:
        per_class = {}
        for class_id in range(NUM_CLASSES):
            eligible = []
            for key, record in records.items():
                count = record["counts"][class_id]
                percent = 100.0 * count / record["pixels"]
                if count >= args.min_class_pixels and percent >= args.min_class_percent:
                    eligible.append(key)
            if len(eligible) < max(SHOTS):
                raise ValueError(
                    f"Class {class_id} {CLASS_NAMES[class_id]} has only {len(eligible)} eligible images; "
                    "lower the selection thresholds"
                )
            eligible.sort()
            random.Random(f"{args.selection_seed}:{class_id}").shuffle(eligible)
            per_class[str(class_id)] = eligible[:max(SHOTS)]

    for class_id in range(NUM_CLASSES):
        for key in per_class[str(class_id)]:
            if key not in records:
                raise ValueError(f"Selected support is absent from train_val: {key}")
            if records[key]["counts"][class_id] == 0:
                raise ValueError(f"Selected support {key} does not contain class {class_id}")

    reserved = list(dict.fromkeys(key for values in per_class.values() for key in values))
    evaluation = [sample.key for sample in test]
    selection_stats = {
        str(class_id): {
            key: {
                "pixels": records[key]["counts"][class_id],
                "percent": 100.0 * records[key]["counts"][class_id] / records[key]["pixels"],
            }
            for key in per_class[str(class_id)]
        }
        for class_id in range(NUM_CLASSES)
    }
    write_json(output / "selection.json", {
        "selection_seed": None if args.selection_file else args.selection_seed,
        "selection_file": str(args.selection_file.resolve()) if args.selection_file else None,
        "thresholds": {"min_class_pixels": args.min_class_pixels,
                       "min_class_percent": args.min_class_percent},
        "per_class": per_class,
        "selection_stats": selection_stats,
    })
    write_json(output / "dataset_audit.json", {
        "train_images": len(train), "test_images": len(test), "train_pixels_after_size_repair": total_pixels,
        "size_repaired_masks": repaired_masks, "off_palette_masks": off_palette_masks,
        "off_palette_pixels": off_palette_pixels, "far_from_palette_pixels": far_pixels,
        "class_image_presence": dict(zip(CLASS_NAMES, image_presence.tolist())),
        "class_pixels": dict(zip(CLASS_NAMES, total_histogram.tolist())),
        "class_pixel_percent": dict(zip(
            CLASS_NAMES, (total_histogram / max(1, total_histogram.sum()) * 100).tolist()
        )),
    })
    for k in SHOTS:
        selected_by_class = {str(c): per_class[str(c)][:k] for c in range(NUM_CLASSES)}
        samples = list(dict.fromkeys(key for values in selected_by_class.values() for key in values))
        write_json(output / f"{k}shot.json", {
            "protocol": "K ordered train_val images per SUIM class; full masks; union deduplicated",
            "split": "train_val", "evaluation_split": "TEST", "shots_per_class": k,
            "target_classes": list(range(NUM_CLASSES)), "classes": class_metadata(), "ignore_index": None,
            "per_class": selected_by_class, "samples": samples, "num_unique_images": len(samples),
            "evaluation_samples": evaluation, "reserved_support_samples": reserved,
            "class_presence": {
                key: [c for c, count in enumerate(records[key]["counts"]) if count]
                for key in samples
            },
        })
    write_json(output / "evaluation.json", {
        "split": "TEST", "protocol": "Fixed official SUIM TEST; never used for support selection",
        "samples": evaluation,
    })
    manifests, checked_evaluation, fingerprint = load_protocol(output)
    print(json.dumps({
        "protocol_dir": str(output), "protocol_sha256": fingerprint,
        "support_images": {str(k): len(manifests[k]["samples"]) for k in SHOTS},
        "evaluation_images": len(checked_evaluation),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()

