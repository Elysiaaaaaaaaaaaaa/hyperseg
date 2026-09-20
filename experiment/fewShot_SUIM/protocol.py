"""Create, validate, and resolve fixed nested SUIM few-shot manifests."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from experiment.fewShot_SUIM.constants import CLASS_NAMES, NUM_CLASSES


SHOTS = (1, 2, 5, 10)


def digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _keys(values: object, split: str) -> set[str]:
    if not isinstance(values, list) or any(not isinstance(value, str) for value in values):
        raise ValueError("Expected a list of sample keys")
    if len(values) != len(set(values)):
        raise ValueError("Duplicate sample keys")
    result = set()
    for value in values:
        parts = value.split("/")
        if (len(parts) != 2 or parts[0].lower() != split.lower()
                or Path(parts[1]).name != parts[1]
                or Path(parts[1]).suffix.lower() not in (".jpg", ".jpeg")):
            raise ValueError(f"Invalid {split} sample key: {value}")
        result.add(value)
    return result


def load_protocol(directory: Path):
    directory = Path(directory)
    manifests = {k: json.loads((directory / f"{k}shot.json").read_text(encoding="utf-8")) for k in SHOTS}
    evaluation_payload = json.loads((directory / "evaluation.json").read_text(encoding="utf-8"))
    evaluation = evaluation_payload["samples"]
    evaluation_set = _keys(evaluation, "TEST")
    if evaluation_payload.get("split") != "TEST" or len(evaluation) != 110:
        raise ValueError("Evaluation must be the complete official 110-image TEST split")

    target_classes = list(range(NUM_CLASSES))
    maximum = manifests[max(SHOTS)]
    reserved = _keys(maximum["samples"], "train_val")
    for k, manifest in manifests.items():
        if (manifest.get("split") != "train_val" or manifest.get("shots_per_class") != k
                or manifest.get("target_classes") != target_classes
                or manifest.get("ignore_index", "missing") is not None):
            raise ValueError(f"{k}-shot metadata mismatch")
        if _keys(manifest["evaluation_samples"], "TEST") != evaluation_set:
            raise ValueError("All K must share the official TEST evaluation set")
        if _keys(manifest["reserved_support_samples"], "train_val") != reserved:
            raise ValueError("All K must share the maximum reserved support set")
        per_class = manifest["per_class"]
        if set(per_class) != {str(c) for c in target_classes}:
            raise ValueError("Per-class keys must be exactly 0..7")
        union = []
        for class_id in target_classes:
            selected = per_class[str(class_id)]
            _keys(selected, "train_val")
            if len(selected) != k or selected != maximum["per_class"][str(class_id)][:k]:
                raise ValueError(f"Class {class_id}: supports must be nested ordered prefixes")
            union.extend(selected)
        if _keys(manifest["samples"], "train_val") != set(union):
            raise ValueError(f"{k}-shot sample union does not match per-class selections")
    return manifests, evaluation, digest({"manifests": manifests, "evaluation": evaluation})


def resolve_protocol_paths(root: Path, keys: list[str]) -> dict[str, tuple[Path, Path]]:
    root = Path(root)
    result = {}
    for key in keys:
        parts = key.split("/")
        if len(parts) != 2 or parts[0].lower() not in ("train_val", "test"):
            raise ValueError(f"Invalid SUIM sample key: {key}")
        split = "train_val" if parts[0].lower() == "train_val" else "TEST"
        name = parts[1]
        if Path(name).name != name or Path(name).suffix.lower() not in (".jpg", ".jpeg"):
            raise ValueError(f"Invalid SUIM image name: {key}")
        image = root / split / "images" / name
        mask = root / split / "masks" / f"{Path(name).stem}.bmp"
        if not image.is_file() or not mask.is_file():
            raise FileNotFoundError(f"Missing image/mask pair for {key}: {image}, {mask}")
        result[key] = image, mask
    return result


def resolve_protocol_samples(root: Path, manifest: dict, evaluation: list[str]):
    from experiment.fewShot_SUIM.data import resolve_sample

    assigned: dict[str, list[int]] = {key: [] for key in manifest["samples"]}
    for class_text, keys in manifest["per_class"].items():
        for key in keys:
            assigned[key].append(int(class_text))
    support = [resolve_sample(root, key, tuple(assigned[key])) for key in manifest["samples"]]
    test = [resolve_sample(root, key) for key in evaluation]
    return support, test


def class_metadata() -> dict[str, str]:
    return {str(index): name for index, name in enumerate(CLASS_NAMES)}
