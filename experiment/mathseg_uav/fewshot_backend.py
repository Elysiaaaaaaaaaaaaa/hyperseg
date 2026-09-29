"""MathSeg transfer and update scopes for the shared SUIM few-shot protocol."""


def load_mathseg(checkpoint, backbone_path=None, head_init="random", dataset="suim", label_policy="standard"):
    import torch
    from experiment.mathseg_uav.model import MathSegUAV, VARIANTS
    from experiment.fewShot_SUIM.run import SEMANTIC_MAPPING

    if dataset not in ("suim", "loveda"):
        raise ValueError(f"Unsupported dataset: {dataset}")
    if dataset == "loveda" and head_init != "semantic-map":
        raise ValueError("LoveDA fixed protocol requires semantic-map initialization")
    config = dict(checkpoint["model_config"])
    if config.get("variant") not in VARIANTS or config.get("classes", 9) != 9:
        raise ValueError("Expected a 9-class MathSeg UAV checkpoint")
    if head_init not in ("random", "semantic-map"):
        raise ValueError(f"Unsupported head initialization: {head_init}")
    # Strictly validate the complete source checkpoint before replacing its head.
    config.update(pretrained=False, classes=9)
    if backbone_path:
        config["model_name"] = str(backbone_path)
    model = MathSegUAV(**config)
    model.load_state_dict(checkpoint["model"], strict=True)
    source = model.head.segmentation
    if dataset == "loveda" and label_policy != "standard":
        raise ValueError("LoveDA requires the standard seven-class protocol")
    classes = 7 if dataset == "loveda" else 8
    target = torch.nn.Conv2d(source.in_channels, classes, 1)
    mapping = ({c: c + 1 for c in range(7)} if dataset == "loveda"
               else SEMANTIC_MAPPING if head_init == "semantic-map" else {})
    with torch.no_grad():
        for target_id, source_id in mapping.items():
            target.weight[target_id].copy_(source.weight[source_id])
            target.bias[target_id].copy_(source.bias[source_id])
    model.head.segmentation = target
    model.model_config["classes"] = classes
    return model, {
        "model": "mathseg", "dataset": dataset, "variant": model.variant, "head_init": head_init,
        "head_mapping_target_to_source": {str(k): v for k, v in mapping.items()},
        "head_random_channels": [c for c in range(classes) if c not in mapping],
        "loaded_non_head_tensors": len(checkpoint["model"]) - 2,
    }


def configure_mathseg(model, mode):
    if mode not in ("semantic-head", "head", "adapter", "full"):
        raise ValueError(f"Unsupported training mode: {mode}")
    for parameter in model.parameters():
        parameter.requires_grad = False
    if mode == "head":
        for module in (model.head.segmentation, model.head.boundary):
            for parameter in module.parameters():
                parameter.requires_grad = True
        return
    module = {"semantic-head": model.head.segmentation, "adapter": model.head, "full": model}[mode]
    for parameter in module.parameters():
        parameter.requires_grad = True


def checkpoint_payload(model, step, info, source_path, source_sha256, checkpoint_format="full"):
    """Store frozen-backbone adaptations without duplicating the source encoder."""
    state = model.state_dict()
    if checkpoint_format == "compact":
        if any(p.requires_grad for p in model.encoder.parameters()):
            raise ValueError("Compact checkpoints require a frozen encoder")
        state = {key: value for key, value in state.items() if not key.startswith("encoder.")}
    elif checkpoint_format != "full":
        raise ValueError(checkpoint_format)
    return dict(model=state, model_config=model.model_config, step=step, experiment=info,
                format=f"mathseg_{checkpoint_format}_v1", source_checkpoint=str(source_path),
                source_sha256=source_sha256)


def restore_compact(payload, source_path=None):
    """Restore an adapted MathSeg model using its verified original UAV encoder."""
    import torch
    from pathlib import Path
    from experiment.mathseg_uav.model import MathSegUAV
    from experiment.fewShot_SUIM.run import file_sha256

    if payload.get("format") != "mathseg_compact_v1":
        raise ValueError("Expected a compact MathSeg checkpoint")
    path = Path(source_path or payload["source_checkpoint"])
    if file_sha256(path) != payload["source_sha256"]:
        raise ValueError("Source checkpoint SHA-256 mismatch")
    source = torch.load(path, map_location="cpu", weights_only=False)
    state = {key: value for key, value in source["model"].items() if key.startswith("encoder.")}
    if any(key.startswith("encoder.") for key in payload["model"]):
        raise ValueError("Compact checkpoint unexpectedly contains encoder tensors")
    state.update(payload["model"])
    model = MathSegUAV(**payload["model_config"])
    model.load_state_dict(state, strict=True)
    return model
