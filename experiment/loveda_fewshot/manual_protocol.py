"""Validate the GUI fixed, nested per-class support protocol without PyTorch."""
import hashlib
import json
from pathlib import Path

SHOTS = (0, 1, 2, 5, 10)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def load_protocol(directory):
    directory = Path(directory)
    manifests = {k: json.loads((directory / f"{k}shot.json").read_text(encoding="utf-8")) for k in SHOTS}
    evaluation = json.loads((directory / "evaluation.json").read_text(encoding="utf-8"))["samples"]

    def keys(values):
        if not isinstance(values, list) or any(not isinstance(v, str) for v in values):
            raise ValueError("Expected a list of sample keys")
        if len(set(values)) != len(values):
            raise ValueError("Duplicate sample keys")
        for value in values:
            parts = value.split("/")
            if (len(parts) != 2 or parts[0] not in ("urban", "rural") or
                    not parts[1].endswith(".png") or "\\" in value or ":" in value):
                raise ValueError(f"Invalid domain/filename key: {value}")
        return set(values)

    eval_set = keys(evaluation)
    if not evaluation:
        raise ValueError("Evaluation set is empty")
    classes = manifests[10]["target_classes"]
    if not classes or any(type(c) is not int or c not in range(1, 8) for c in classes) or len(set(classes)) != len(classes):
        raise ValueError("Invalid target classes")
    reserved = keys(manifests[10]["samples"])
    if reserved & eval_set:
        raise ValueError("Support/evaluation leakage")
    for k, manifest in manifests.items():
        if manifest["split"].lower() != "val" or manifest["shots_per_class"] != k or manifest["target_classes"] != classes:
            raise ValueError(f"{k}-shot metadata mismatch")
        if keys(manifest["evaluation_samples"]) != eval_set or keys(manifest["reserved_support_samples"]) != reserved:
            raise ValueError("All K must share the same evaluation and reserved support sets")
        per_class = manifest["per_class"]
        if set(per_class) != {str(c) for c in classes}:
            raise ValueError("Per-class keys do not match targets")
        union = []
        for c in classes:
            chosen = per_class[str(c)]
            keys(chosen)
            if len(chosen) != k or chosen != manifests[10]["per_class"][str(c)][:k]:
                raise ValueError(f"Class {c}: supports must be ordered nested prefixes")
            union.extend(chosen)
        if keys(manifest["samples"]) != set(union):
            raise ValueError("Support list does not match per-class union")
    return manifests, evaluation, digest({"manifests": manifests, "evaluation": evaluation})


def resolve_samples(root, sample_keys):
    root = Path(root)
    def child(parent, wanted):
        matches = [p for p in parent.iterdir() if p.is_dir() and p.name.lower() == wanted.lower()]
        if len(matches) != 1:
            raise ValueError(f"Expected {wanted} directory under {parent}")
        return matches[0]
    val = root if root.name.lower() == "val" else child(root, "val")
    domains = {}
    for domain in ("urban", "rural"):
        folder = child(val, domain)
        domains[domain] = child(folder, "images_png"), child(folder, "masks_png")
    result = {}
    for key in sample_keys:
        domain, name = key.split("/")
        image_dir, mask_dir = domains[domain]
        image, mask = image_dir / name, mask_dir / name
        if not image.is_file() or not mask.is_file():
            raise FileNotFoundError(f"Missing image/mask: {key}")
        result[key] = (image, mask)
    return result
