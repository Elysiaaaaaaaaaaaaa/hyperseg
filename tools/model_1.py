"""
HyperSeg-UAV v2
SegFormer-B3 + Residual Dynamic Modulation
+ Fixed Low-Rank Adapter + Cross-scale Fusion
+ Boundary Refinement

Compatible with the original training interface:
    model = HyperSegUAV(pretrained=True)
    output = model(image)
    output["logits"]
    output["boundary"]
    output["context"]

Requirements:
    pip install torch transformers
"""

import torch
import torch.nn.functional as F
from torch import nn


# ============================================================
# 1. Basic MLP
# ============================================================

class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden=None):
        super().__init__()

        hidden = hidden or max(in_dim, out_dim)

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x):
        return self.net(x)


# ============================================================
# 2. SceneCondition
# ============================================================

class SceneCondition(nn.Module):
    """
    Extract global scene context from multi-scale features.

    Input:
        F1: [B, 64,  H/4,  W/4]
        F2: [B, 128, H/8,  W/8]
        F3: [B, 320, H/16, W/16]
        F4: [B, 512, H/32, W/32]

    Output:
        context: [B, context_dim]
    """

    def __init__(self, channels, dim=128):
        super().__init__()

        self.projections = nn.ModuleList([
            nn.Conv2d(c, dim, kernel_size=1)
            for c in channels
        ])

        self.mlp = MLP(
            dim * len(channels),
            dim,
            hidden=dim,
        )

    def forward(self, features):
        pooled = []

        for projection, feature in zip(
            self.projections,
            features,
        ):
            x = projection(feature)

            x = F.adaptive_avg_pool2d(
                x,
                output_size=1,
            )

            x = x.flatten(1)

            pooled.append(x)

        context = torch.cat(pooled, dim=1)

        return self.mlp(context)


# ============================================================
# 3. Residual Dynamic Modulation
# ============================================================

class ResidualDynamicModulation(nn.Module):
    """
    Scene-conditioned residual feature modulation.

    y = x + alpha * (GN(x) * (1 + gamma) + beta - GN(x))

    The modulation starts from an identity mapping.
    """

    def __init__(
        self,
        context_dim,
        channels,
        scale=0.1,
    ):
        super().__init__()

        self.scale = scale

        self.layers = nn.ModuleList([
            nn.Linear(context_dim, 2 * c)
            for c in channels
        ])

        self.norms = nn.ModuleList([
            nn.GroupNorm(1, c)
            for c in channels
        ])

        # Zero initialization:
        # gamma = 0, beta = 0
        # therefore y = x at initialization.
        for layer in self.layers:
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, features, context):
        result = []

        for feature, layer, norm in zip(
            features,
            self.layers,
            self.norms,
        ):
            gamma, beta = layer(context).chunk(2, dim=1)

            gamma = self.scale * torch.tanh(
                gamma
            )[:, :, None, None]

            beta = self.scale * torch.tanh(
                beta
            )[:, :, None, None]

            normalized = norm(feature)

            adapted = normalized * (1 + gamma) + beta

            # Residual modulation
            result.append(
                feature + adapted - normalized
            )

        return result


# ============================================================
# 4. Fixed Low-Rank Adapter
# ============================================================

class FixedLowRankAdapter(nn.Module):
    """
    Fixed low-rank adapter with scene-conditioned scaling.

    y = x + scale * B(A(x)) * alpha(context)

    A: fixed learnable down projection
    B: fixed learnable up projection
    alpha: scene-conditioned rank-wise gate

    This avoids generating full LoRA matrices for every sample.
    """

    def __init__(
        self,
        context_dim,
        channels,
        rank=8,
        scale=0.1,
    ):
        super().__init__()

        self.channels = channels
        self.rank = rank
        self.scale = scale

        # Fixed low-rank projections
        self.down = nn.Conv2d(
            channels,
            rank,
            kernel_size=1,
            bias=False,
        )

        self.up = nn.Conv2d(
            rank,
            channels,
            kernel_size=1,
            bias=False,
        )

        # Scene-conditioned rank gate
        self.gate = nn.Sequential(
            nn.Linear(context_dim, rank),
            nn.GELU(),
            nn.Linear(rank, rank),
        )

        # LoRA-style initialization
        nn.init.kaiming_uniform_(
            self.down.weight,
            a=5 ** 0.5,
        )

        # Start from identity
        nn.init.zeros_(self.up.weight)

        nn.init.zeros_(self.gate[-1].weight)
        nn.init.zeros_(self.gate[-1].bias)

    def forward(self, feature, context):
        # [B, C, H, W] -> [B, r, H, W]
        latent = self.down(feature)

        # [B, r] -> [B, r, 1, 1]
        alpha = torch.tanh(
            self.gate(context)
        )[:, :, None, None]

        latent = latent * alpha

        residual = self.up(latent)

        return feature + self.scale * residual


# ============================================================
# 5. Cross-scale Fusion
# ============================================================

class DynamicFusionV2(nn.Module):
    """
    Cross-scale fusion with scene-conditioned channel gating.

    Each feature is projected to out_channels, upsampled to
    the highest-resolution scale, concatenated, and fused.

    Output:
        [B, out_channels, H/4, W/4]
    """

    def __init__(
        self,
        channels,
        context_dim,
        out_channels=64,
    ):
        super().__init__()

        self.out_channels = out_channels

        self.projections = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(
                    c,
                    out_channels,
                    kernel_size=1,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
                nn.GELU(),
            )
            for c in channels
        ])

        self.fuse = nn.Sequential(
            nn.Conv2d(
                out_channels * len(channels),
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),

            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
        )

        # Scene-conditioned channel gate
        self.gate = nn.Sequential(
            nn.Linear(context_dim, out_channels),
            nn.GELU(),
            nn.Linear(out_channels, out_channels),
        )

    def forward(self, features, context, output_size):
        projected = []

        for feature, projection in zip(
            features,
            self.projections,
        ):
            x = projection(feature)

            x = F.interpolate(
                x,
                size=output_size,
                mode="bilinear",
                align_corners=False,
            )

            projected.append(x)

        # Cross-scale interaction
        fused = self.fuse(
            torch.cat(projected, dim=1)
        )

        # Scene-conditioned channel gate
        gate = torch.sigmoid(
            self.gate(context)
        )[:, :, None, None]

        # Residual gating
        return fused * (1.0 + gate)


# ============================================================
# 6. SegFormer-B3 Encoder
# ============================================================

class SegFormerB3Encoder(nn.Module):
    """
    Pretrained Hugging Face SegFormer-B3 feature pyramid.

    Output:
        F1: [B, 64,  H/4,  W/4]
        F2: [B, 128, H/8,  W/8]
        F3: [B, 320, H/16, W/16]
        F4: [B, 512, H/32, W/32]
    """

    def __init__(
        self,
        pretrained=True,
        model_name="nvidia/mit-b3",
    ):
        super().__init__()

        try:
            from transformers import SegformerModel
        except ImportError as error:
            raise ImportError(
                "Please install transformers first:\n"
                "pip install transformers"
            ) from error

        if pretrained:
            self.backbone = SegformerModel.from_pretrained(
                model_name
            )
        else:
            self.backbone = SegformerModel.from_pretrained(
                model_name,
                local_files_only=True,
            )

        self.channels = [64, 128, 320, 512]

    def forward(self, image):
        output = self.backbone(
            pixel_values=image,
            output_hidden_states=True,
        )

        # SegFormer hidden states are channel-last:
        # [B, HW, C]
        features = []

        for hidden, channel in zip(
            output.hidden_states[-4:],
            self.channels,
        ):
            if hidden.ndim == 4:
                features.append(hidden)
                continue

            batch, tokens, hidden_channels = hidden.shape

            # SegFormer feature maps are square for square input.
            height = width = int(tokens ** 0.5)

            if height * width != tokens:
                raise RuntimeError(
                    f"Unexpected SegFormer token shape: "
                    f"{hidden.shape}"
                )

            feature = hidden.transpose(1, 2).reshape(
                batch,
                hidden_channels,
                height,
                width,
            )

            features.append(feature)

        return features


# ============================================================
# 7. Boundary Refinement
# ============================================================

class BoundaryRefinement(nn.Module):
    """
    Lightweight boundary refinement branch.
    """

    def __init__(self, channels):
        super().__init__()

        self.refine = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(channels),
            nn.GELU(),

            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(channels),
            nn.GELU(),
        )

    def forward(self, x):
        return self.refine(x)


# ============================================================
# 8. HyperSeg-UAV v2
# ============================================================

class HyperSegUAV(nn.Module):
    """
    HyperSeg-UAV v2.

    Main improvements:
        1. Residual dynamic modulation
        2. Fixed low-rank adapter
        3. Cross-scale fusion
        4. Boundary refinement

    Forward output:
        {
            "logits": [B, classes, H, W],
            "boundary": [B, 1, H, W],
            "context": [B, context_dim],
        }
    """

    def __init__(
        self,
        classes=9,
        widths=(64, 128, 320, 512),
        context_dim=128,
        pretrained=True,
        model_name="nvidia/mit-b3",
        adapter_rank=8,
        adapter_scale=0.1,
    ):
        super().__init__()

        self.model_config = {
            "classes": classes,
            "widths": list(widths),
            "context_dim": context_dim,
            "pretrained": False,
            "model_name": model_name,
            "adapter_rank": adapter_rank,
            "adapter_scale": adapter_scale,
            "version": "v2",
        }

        # Encoder
        self.encoder = SegFormerB3Encoder(
            pretrained=pretrained,
            model_name=model_name,
        )

        # Scene context
        self.scene = SceneCondition(
            widths,
            context_dim,
        )

        # Residual dynamic modulation
        self.modulation = ResidualDynamicModulation(
            context_dim,
            widths,
            scale=0.1,
        )

        # Cross-scale fusion
        self.fusion = DynamicFusionV2(
            widths,
            context_dim,
            widths[0],
        )

        # Fixed LoRA on fused feature
        self.low_rank = FixedLowRankAdapter(
            context_dim,
            widths[0],
            rank=adapter_rank,
            scale=adapter_scale,
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Conv2d(
                widths[0],
                widths[0],
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(widths[0]),
            nn.GELU(),

            nn.Conv2d(
                widths[0],
                widths[0],
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(widths[0]),
            nn.GELU(),
        )

        # Boundary refinement
        self.boundary_refine = BoundaryRefinement(
            widths[0]
        )

        # Segmentation head
        self.head = nn.Conv2d(
            widths[0],
            classes,
            kernel_size=1,
        )

        # Boundary prediction head
        self.boundary = nn.Conv2d(
            widths[0],
            1,
            kernel_size=1,
        )

    def forward(self, image):
        input_size = image.shape[-2:]

        # ----------------------------------------------------
        # 1. Encoder
        # ----------------------------------------------------
        features = self.encoder(image)

        # ----------------------------------------------------
        # 2. Scene condition
        # ----------------------------------------------------
        context = self.scene(features)

        # ----------------------------------------------------
        # 3. Residual dynamic modulation
        # ----------------------------------------------------
        features = self.modulation(
            features,
            context,
        )

        # ----------------------------------------------------
        # 4. Cross-scale fusion
        # ----------------------------------------------------
        fused = self.fusion(
            features,
            context,
            output_size=features[0].shape[-2:],
        )

        # ----------------------------------------------------
        # 5. Fixed LoRA + dynamic scaling
        # ----------------------------------------------------
        fused = self.low_rank(
            fused,
            context,
        )

        # ----------------------------------------------------
        # 6. Boundary refinement
        # ----------------------------------------------------
        boundary_feature = self.boundary_refine(fused)

        # ----------------------------------------------------
        # 7. Main decoder
        # ----------------------------------------------------
        decoded = self.decoder(fused)

        # Small residual boundary enhancement
        decoded = decoded + 0.1 * boundary_feature

        # ----------------------------------------------------
        # 8. Segmentation output
        # ----------------------------------------------------
        logits = self.head(decoded)

        logits = F.interpolate(
            logits,
            size=input_size,
            mode="bilinear",
            align_corners=False,
        )

        # ----------------------------------------------------
        # 9. Boundary output
        # ----------------------------------------------------
        boundary = self.boundary(decoded)

        boundary = F.interpolate(
            boundary,
            size=input_size,
            mode="bilinear",
            align_corners=False,
        )

        return {
            "logits": logits,
            "boundary": boundary,
            "context": context,
        }


# ============================================================
# 9. Quick Test
# ============================================================

