"""Swin-L encoder with the repository's HyperSeg decoder path.

The Swin implementation is deliberately imported from the sibling MMSegmentation
checkout.  This keeps the pretrained checkpoint format identical to the
Mask2Former experiment and avoids silently mixing it with a different Swin
implementation.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F

from .model import (
    DynamicFusion,
    DynamicLowRankAdapter,
    DynamicModulation,
    SceneCondition,
)


class SwinLHyperSeg(nn.Module):
    """Swin-L + channel adapters + the original HyperSeg decoder."""

    def __init__(
        self,
        classes: int = 9,
        target_widths=(64, 128, 320, 512),
        context_dim: int = 128,
        adapter_rank: int = 8,
        adapter_scale: float = 0.1,
        pretrained_checkpoint: Optional[str] = None,
        with_cp: bool = False,
    ):
        super().__init__()
        try:
            from mmseg.models.backbones import SwinTransformer
        except ImportError as error:  # pragma: no cover - depends on server env
            raise ImportError(
                "SwinLHyperSeg requires the sibling MMSegmentation checkout "
                "on PYTHONPATH (and its mmcv/mmengine dependencies)."
            ) from error

        target_widths = tuple(int(width) for width in target_widths)
        if len(target_widths) != 4:
            raise ValueError("target_widths must contain four feature widths")
        self.model_config = {
            "classes": classes,
            "target_widths": list(target_widths),
            "context_dim": context_dim,
            "adapter_rank": adapter_rank,
            "adapter_scale": adapter_scale,
            "pretrained_checkpoint": pretrained_checkpoint,
            "with_cp": with_cp,
        }

        self.backbone = SwinTransformer(
            pretrain_img_size=384,
            embed_dims=192,
            window_size=12,
            depths=(2, 2, 18, 2),
            num_heads=(6, 12, 24, 48),
            strides=(4, 2, 2, 2),
            out_indices=(0, 1, 2, 3),
            mlp_ratio=4,
            qkv_bias=True,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=0.3,
            patch_norm=True,
            with_cp=with_cp,
            init_cfg=(
                dict(type="Pretrained", checkpoint=pretrained_checkpoint)
                if pretrained_checkpoint
                else None
            ),
        )
        # Calling init_weights explicitly makes the no-pretrained smoke test
        # deterministic and ensures the checkpoint load happens before train().
        def audit_load(module, incompatible):
            print(f"Swin pretrained missing={incompatible.missing_keys}, "
                  f"unexpected={incompatible.unexpected_keys}", flush=True)
            # Classification pretraining may omit intermediate output norms.
            allowed = {f"norm{i}.{field}" for i in range(4) for field in ("weight", "bias")}
            missing = set(incompatible.missing_keys) - allowed
            if missing:
                raise RuntimeError(f"Incomplete Swin backbone initialization: {sorted(missing)}")

        audit_hook = self.backbone.register_load_state_dict_post_hook(audit_load)
        self.backbone.init_weights()
        audit_hook.remove()
        source_widths = (192, 384, 768, 1536)
        self.projections = nn.ModuleList(
            [nn.Conv2d(src, dst, kernel_size=1) for src, dst in zip(source_widths, target_widths)]
        )

        self.scene = SceneCondition(target_widths, context_dim)
        self.modulation = DynamicModulation(context_dim, target_widths)
        self.fusion = DynamicFusion(target_widths, context_dim, target_widths[0])
        self.low_rank = DynamicLowRankAdapter(
            context_dim, target_widths[0], adapter_rank, adapter_scale
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(target_widths[0], target_widths[0], 3, padding=1, bias=False),
            nn.BatchNorm2d(target_widths[0]),
            nn.GELU(),
            nn.Conv2d(target_widths[0], target_widths[0], 3, padding=1, bias=False),
            nn.BatchNorm2d(target_widths[0]),
            nn.GELU(),
        )
        self.head = nn.Conv2d(target_widths[0], classes, 1)
        self.boundary = nn.Conv2d(target_widths[0], 1, 1)
        self.register_buffer(
            "image_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "image_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

    def forward(self, image: torch.Tensor):
        input_size = image.shape[-2:]
        image = (image - self.image_mean) / self.image_std
        features = [projection(feature) for projection, feature in zip(
            self.projections, self.backbone(image)
        )]
        context = self.scene(features)
        features = self.modulation(features, context)
        fused = self.fusion(features, context, features[0].shape[-2:])
        fused = self.low_rank(fused, context)
        decoded = self.decoder(fused)
        logits = F.interpolate(self.head(decoded), input_size, mode="bilinear", align_corners=False)
        boundary = F.interpolate(
            self.boundary(decoded), input_size, mode="bilinear", align_corners=False
        )
        return {"logits": logits, "boundary": boundary, "context": context}
