"""Load the HyperSeg-UAV v2 checkpoint (``tools/model_1.py``) into a ready model.

``models/hyperseg_resume_best.pt`` was trained with the v2 architecture, not the v1
one in ``hyperseg_uav/model.py``.  Two structural differences make a plain
``HyperSegUAV(**config)`` fail:

* ``model_config`` carries ``"version": "v2"``, which the v2 constructor does not
  accept, so it has to be dropped before construction;
* the state dict holds 730 tensors against v1's 686 -- v2 adds
  ``boundary_refine`` (12), widens ``fusion`` from 12 to 40 and ``low_rank`` from
  2 to 6 -- so a v1 model cannot hold these weights at all.

The forward pass needs the same input convention as the v1 exporter: images scaled
to ``[0, 1]`` and fed straight in, with no ImageNet normalisation, because both
backbones call the Hugging Face SegFormer without a preprocessor.

The Transformers-5 -> Transformers-4 encoder renaming is identical to the v1 path
(``hyperseg_uav.model.legacy_encoder_key``, byte-for-byte the same table as
``experiment/loveda_fewshot/train.py`` used to transfer these weights).
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hyperseg_uav.model import legacy_encoder_key  # noqa: E402


def load_v2(checkpoint: Path, device: str | torch.device = "cpu"):
    """Return ``(model, checkpoint_state, info)`` for a v2 HyperSeg-UAV checkpoint.

    Loading is strict in both directions: every checkpoint tensor must find its
    destination and every destination tensor must be filled, so a silently dropped
    backbone cannot slip through.
    """
    from tools.model_1 import HyperSegUAV

    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    config = dict(state.get("model_config", {}))
    version = config.pop("version", None)
    if version != "v2":
        raise RuntimeError(f"expected a v2 checkpoint, found version={version!r}")
    config["pretrained"] = False

    # Prefer the checkpoint's backbone copy in models/ over the Hub cache: the
    # servers run without outbound access often enough that a Hub fallback is a
    # liability, and the local copy is byte-identical to what was trained against.
    local_backbone = ROOT / "models" / str(config.get("model_name", "")).replace("/", "--")
    if local_backbone.is_dir():
        config["model_name"] = str(local_backbone)

    model = HyperSegUAV(**config)
    destination = model.state_dict()
    mapped: dict[str, torch.Tensor] = {}
    renamed = 0
    for key, tensor in state["model"].items():
        target = key if key in destination else legacy_encoder_key(key)
        if target is None or target not in destination:
            raise KeyError(f"checkpoint tensor {key!r} has no destination in the v2 model")
        if target in mapped:
            raise KeyError(f"two checkpoint tensors map to {target!r}")
        if tuple(tensor.shape) != tuple(destination[target].shape):
            raise ValueError(
                f"shape mismatch for {target}: checkpoint {tuple(tensor.shape)} "
                f"vs model {tuple(destination[target].shape)}"
            )
        mapped[target] = tensor
        renamed += int(target != key)

    missing = [key for key in destination if key not in mapped]
    if missing:
        raise KeyError(f"{len(missing)} model tensors left unfilled, e.g. {missing[:5]}")
    model.load_state_dict(mapped, strict=True)
    model.to(device).eval()

    info = {
        "architecture": "HyperSeg-UAV v2 (tools/model_1.py)",
        "checkpoint": str(checkpoint),
        "checkpoint_epoch": state.get("epoch"),
        "checkpoint_val_miou": state.get("val_miou"),
        "tensors_loaded": len(mapped),
        "translated_legacy_encoder_keys": renamed,
        "model_config": dict(state.get("model_config", {})),
        "backbone": config.get("model_name"),
    }
    return model, state, info
