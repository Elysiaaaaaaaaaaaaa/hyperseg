"""
model_v2.py
Few-shot + HyperNetwork UAV semantic segmentation model.

Main optimization:
1. Support and query images are encoded in ONE SegFormer forward pass.
2. Supports query_size > 1.
3. Keeps the original HyperSegUAV API.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Sequence

try:
    from transformers import SegformerModel, SegformerConfig
except ImportError:
    SegformerModel = None
    SegformerConfig = None

NUM_CLASSES = 9
IGNORE_INDEX = 0
DEFAULT_CLASS_IDS = [2, 3, 4, 5, 6, 7, 8]


class ConvGNAct(nn.Module):
    def __init__(self, in_ch, out_ch, groups=8):
        super().__init__()
        groups = min(groups, out_ch)
        while out_ch % groups:
            groups -= 1
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.GroupNorm(groups, out_ch),
            nn.GELU(),
        )

    def forward(self, x):
        return self.block(x)


class MaskedPrototypeExtractor(nn.Module):
    def __init__(self, feature_dim, prototype_dim=256):
        super().__init__()
        self.project = nn.Sequential(
            nn.Linear(feature_dim, prototype_dim),
            nn.LayerNorm(prototype_dim),
            nn.GELU(),
        )

    def forward(self, support_features, support_masks, class_ids, k_shot):
        B, S, C, h, w = support_features.shape
        N = len(class_ids)
        if S != N * k_shot:
            raise ValueError(f"S={S}, but N*K={N*k_shot}")

        masks = F.interpolate(
            support_masks.float().reshape(B * S, 1, *support_masks.shape[-2:]),
            size=(h, w),
            mode="nearest",
        ).reshape(B, S, h, w).long()

        prototypes = []
        for n, cls in enumerate(class_ids):
            shots = []
            for k in range(k_shot):
                s = n * k_shot + k
                feat = support_features[:, s]
                m = (masks[:, s] == int(cls)).float()
                denom = m.flatten(1).sum(1, keepdim=True)
                pooled = (feat * m[:, None]).flatten(2).sum(-1)
                pooled = pooled / denom.clamp_min(1.0)
                fallback = feat.flatten(2).mean(-1)
                valid = (denom > 0).float()
                pooled = pooled * valid + fallback * (1.0 - valid)
                shots.append(pooled)
            prototypes.append(torch.stack(shots, 1).mean(1))

        prototypes = torch.stack(prototypes, 1)
        return F.normalize(self.project(prototypes), dim=-1)


class HyperNetwork(nn.Module):
    def __init__(self, prototype_dim=256, context_dim=256, num_scales=4):
        super().__init__()
        self.num_scales = num_scales
        self.context = nn.Sequential(
            nn.Linear(prototype_dim, context_dim),
            nn.LayerNorm(context_dim),
            nn.GELU(),
            nn.Linear(context_dim, context_dim),
            nn.LayerNorm(context_dim),
            nn.GELU(),
        )
        self.modulation = nn.Linear(context_dim, num_scales * 2)
        self.fusion = nn.Linear(context_dim, num_scales)
        self.adapter_gate = nn.Linear(context_dim, 1)
        self.boundary_scale = nn.Linear(context_dim, 1)
        self.prototype_scale = nn.Linear(context_dim, 1)

        for m in [
            self.modulation, self.fusion, self.adapter_gate,
            self.boundary_scale, self.prototype_scale
        ]:
            nn.init.zeros_(m.weight)
            nn.init.zeros_(m.bias)
        nn.init.constant_(self.adapter_gate.bias, -2.0)
        nn.init.constant_(self.boundary_scale.bias, -2.0)
        nn.init.constant_(self.prototype_scale.bias, -2.0)

    def forward(self, prototypes):
        context = self.context(prototypes.mean(1))
        raw = self.modulation(context).view(-1, self.num_scales, 2)
        gamma = 1.0 + 0.1 * torch.tanh(raw[..., 0])
        beta = 0.1 * torch.tanh(raw[..., 1])
        return {
            "context": context,
            "gamma": gamma,
            "beta": beta,
            "fusion_weights": F.softmax(self.fusion(context), dim=-1),
            "adapter_gate": torch.sigmoid(self.adapter_gate(context)),
            "boundary_scale": torch.sigmoid(self.boundary_scale(context)),
            "prototype_scale": torch.sigmoid(self.prototype_scale(context)),
        }


class DynamicModulation(nn.Module):
    def __init__(self, channels, context_dim=256):
        super().__init__()
        groups = min(32, channels)
        while channels % groups:
            groups -= 1
        self.norm = nn.GroupNorm(groups, channels)
        self.gamma = nn.Linear(context_dim, channels)
        self.beta = nn.Linear(context_dim, channels)
        nn.init.zeros_(self.gamma.weight)
        nn.init.zeros_(self.gamma.bias)
        nn.init.zeros_(self.beta.weight)
        nn.init.zeros_(self.beta.bias)

    def forward(self, x, context, ext_gamma, ext_beta):
        """
        x:          [Bq, C, H, W]
        context:    [Bq, context_dim]
        ext_gamma:  [Bq] or [Bq, 1]
        ext_beta:   [Bq] or [Bq, 1]
        """
        
        Bq = x.shape[0]
        
        # ---------------------------------------------------------
        # 1. 强制保证 context 的 batch 维度与 x 一致
        # ---------------------------------------------------------
        if context.ndim == 1:
            context = context.unsqueeze(0)
        
        if context.shape[0] != Bq:
            raise RuntimeError(
                f"[DynamicModulation] context batch mismatch: "
                f"x={tuple(x.shape)}, context={tuple(context.shape)}"
            )
        
        # ---------------------------------------------------------
        # 2. 处理外部 gamma
        # ---------------------------------------------------------
        if ext_gamma.ndim == 0:
            ext_gamma = ext_gamma.expand(Bq)
        
        elif ext_gamma.ndim == 1:
            if ext_gamma.shape[0] == 1:
                ext_gamma = ext_gamma.expand(Bq)
            elif ext_gamma.shape[0] != Bq:
                raise RuntimeError(
                    f"[DynamicModulation] gamma batch mismatch: "
                    f"x={tuple(x.shape)}, gamma={tuple(ext_gamma.shape)}"
                )
        
        else:
            # [Bq, 1] -> [Bq]
            # [Bq, ...] -> 对剩余维度取平均
            if ext_gamma.shape[0] != Bq:
                raise RuntimeError(
                    f"[DynamicModulation] gamma batch mismatch: "
                    f"x={tuple(x.shape)}, gamma={tuple(ext_gamma.shape)}"
                )
        
            ext_gamma = ext_gamma.reshape(Bq, -1).mean(dim=1)
        
        # ---------------------------------------------------------
        # 3. 处理外部 beta
        # ---------------------------------------------------------
        if ext_beta.ndim == 0:
            ext_beta = ext_beta.expand(Bq)
        
        elif ext_beta.ndim == 1:
            if ext_beta.shape[0] == 1:
                ext_beta = ext_beta.expand(Bq)
            elif ext_beta.shape[0] != Bq:
                raise RuntimeError(
                    f"[DynamicModulation] beta batch mismatch: "
                    f"x={tuple(x.shape)}, beta={tuple(ext_beta.shape)}"
                )
        
        else:
            if ext_beta.shape[0] != Bq:
                raise RuntimeError(
                    f"[DynamicModulation] beta batch mismatch: "
                    f"x={tuple(x.shape)}, beta={tuple(ext_beta.shape)}"
                )
        
            ext_beta = ext_beta.reshape(Bq, -1).mean(dim=1)
        
        # ---------------------------------------------------------
        # 4. 保证 dtype/device 一致
        # ---------------------------------------------------------
        ext_gamma = ext_gamma.to(device=x.device, dtype=x.dtype)
        ext_beta = ext_beta.to(device=x.device, dtype=x.dtype)
        
        # ---------------------------------------------------------
        # 5. Dynamic modulation
        # ---------------------------------------------------------
        y = self.norm(x)
        
        g = 1.0 + 0.1 * torch.tanh(self.gamma(context))
        b = self.beta(context)
        
        y = y * g.unsqueeze(-1).unsqueeze(-1)
        y = y + b.unsqueeze(-1).unsqueeze(-1)
        
        # 外部 task-specific modulation
        y = (
            y * ext_gamma.view(Bq, 1, 1, 1)
            + ext_beta.view(Bq, 1, 1, 1)
        )
        
        return x + 0.1 * y


class LowRankAdapter(nn.Module):
    def __init__(self, channels, rank=32, context_dim=256):
        super().__init__()
        self.down = nn.Conv2d(channels, rank, 1, bias=False)
        self.up = nn.Conv2d(rank, channels, 1, bias=False)
        self.gate = nn.Linear(context_dim, 1)
        nn.init.kaiming_normal_(self.down.weight, mode="fan_out")
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, -2.0)

    def forward(self, x, context, external_gate):
        z = self.up(F.gelu(self.down(x)))
        gate = torch.sigmoid(self.gate(context)) * external_gate
        return x + gate[..., None, None] * z


class DynamicFusion(nn.Module):
    def __init__(self, channels=(64, 128, 320, 512), dim=128, context_dim=256):
        super().__init__()
        self.proj = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(c, dim, 1, bias=False),
                nn.GroupNorm(16, dim),
                nn.GELU(),
            )
            for c in channels
        ])
        self.refine = nn.Sequential(
            ConvGNAct(dim, dim),
            ConvGNAct(dim, dim),
        )
        self.channel_gate = nn.Linear(context_dim, dim)

    def forward(self, features, context, weights):
        size = features[0].shape[-2:]
        fused = 0.0
        for i, x in enumerate(features):
            x = self.proj[i](x)
            x = F.interpolate(x, size=size, mode="bilinear", align_corners=False)
            fused = fused + weights[:, i, None, None, None] * x
        fused = self.refine(fused)
        gate = torch.sigmoid(self.channel_gate(context))[..., None, None]
        return fused * (0.5 + gate)


class Decoder(nn.Module):
    def __init__(self, dim=128, prototype_dim=256, num_classes=9):
        super().__init__()

        self.block1 = nn.Sequential(
            ConvGNAct(dim, 128),
            ConvGNAct(128, 128),
        )

        self.block2 = nn.Sequential(
            ConvGNAct(128, 96),
            ConvGNAct(96, 96),
        )

        # Prototype -> 与 block1 输出保持一致的 128 channels
        self.proto = nn.Sequential(
            nn.Linear(prototype_dim, 96),
            nn.GELU(),
            nn.Linear(96, 128),
        )

        self.head = nn.Conv2d(96, num_classes, 1)

        self.proto_scale = nn.Parameter(
            torch.tensor(0.1)
        )

    def forward(self, x, prototypes, scale):
        x = self.block1(x)

        # x: [B*Q, 128, H, W]
        # p: [B*Q, 128, 1, 1]
        p = self.proto(prototypes.mean(1))[..., None, None]

        x = (
            x
            + self.proto_scale
            * scale[..., None, None]
            * p
        )

        x = F.interpolate(
            x,
            scale_factor=2,
            mode="bilinear",
            align_corners=False,
        )

        x = self.block2(x)

        x = F.interpolate(
            x,
            scale_factor=2,
            mode="bilinear",
            align_corners=False,
        )

        return self.head(x)


class BoundaryHead(nn.Module):
    def __init__(self, dim=128, context_dim=256):
        super().__init__()
        self.body = nn.Sequential(
            ConvGNAct(dim, 64),
            ConvGNAct(64, 32),
            nn.Conv2d(32, 1, 1),
        )
        self.scale = nn.Linear(context_dim, 1)
        nn.init.zeros_(self.scale.weight)
        nn.init.constant_(self.scale.bias, -2.0)

    def forward(self, x, context, task_scale):
        y = self.body(x)
        s = torch.sigmoid(self.scale(context)) * task_scale
        return y * (0.5 + s[..., None, None])


class HyperSegUAV(nn.Module):
    """
    Optimized Few-shot + HyperNetwork segmentation.

    Inputs:
        support_images: [B, S, 3, H, W], S=N*K
        support_masks:  [B, S, H, W]
        query_images:   [B, Q, 3, H, W] or [B, 3, H, W]

    Outputs:
        logits: [B, Q, 9, H, W] when Q>1
                [B, 9, H, W] when Q=1
    """
    def __init__(
        self,
        num_classes=9,
        class_ids: Sequence[int] = DEFAULT_CLASS_IDS,
        k_shot=1,
        backbone_name="nvidia/mit-b3",
        pretrained=True,
        prototype_dim=256,
        context_dim=256,
        fusion_dim=128,
        adapter_rank=32,
        freeze_backbone=False,
    ):
        super().__init__()
        if SegformerModel is None:
            raise ImportError("Install transformers: pip install transformers")

        self.num_classes = num_classes
        self.class_ids = list(map(int, class_ids))
        self.k_shot = k_shot
        self.feature_channels = [64, 128, 320, 512]

        if pretrained:
            self.backbone = SegformerModel.from_pretrained(backbone_name)
        else:
            cfg = SegformerConfig.from_pretrained(backbone_name)
            self.backbone = SegformerModel(cfg)

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        self.support_proj = nn.Conv2d(64, prototype_dim, 1)
        self.prototype_extractor = MaskedPrototypeExtractor(
            prototype_dim, prototype_dim
        )

        self.hyper = HyperNetwork(prototype_dim, context_dim, 4)
        self.modulation = nn.ModuleList([
            DynamicModulation(c, context_dim)
            for c in self.feature_channels
        ])
        self.adapters = nn.ModuleList([
            LowRankAdapter(c, adapter_rank, context_dim)
            for c in self.feature_channels
        ])
        self.fusion = DynamicFusion(
            self.feature_channels, fusion_dim, context_dim
        )
        self.decoder = Decoder(fusion_dim, prototype_dim, num_classes)
        self.boundary = BoundaryHead(fusion_dim, context_dim)

    def _encode(self, images):
        out = self.backbone(images, output_hidden_states=True)
        states = list(out.hidden_states[-4:])
        features = []
        for x in states:
            if x.shape[1] not in self.feature_channels and x.shape[-1] in self.feature_channels:
                x = x.permute(0, 3, 1, 2).contiguous()
            features.append(x)
        return features

    def forward(self, query_images, support_images=None, support_masks=None,
                return_features=False):
        if support_images is None or support_masks is None:
            raise ValueError("support_images and support_masks are required")

        if query_images.ndim == 4:
            query_images = query_images.unsqueeze(1)
        if support_images.ndim != 5 or support_masks.ndim != 4:
            raise ValueError(
                "Expected support_images [B,S,3,H,W], "
                "support_masks [B,S,H,W], query_images [B,Q,3,H,W]"
            )

        B, S, C, H, W = support_images.shape
        Bq, Q, Cq, Hq, Wq = query_images.shape
        if B != Bq:
            raise ValueError("Support/query batch sizes differ")

        # Critical GPU-utilization optimization:
        # encode all support + query images in ONE backbone call.
        all_images = torch.cat(
            [support_images.reshape(B * S, C, H, W),
             query_images.reshape(B * Q, Cq, Hq, W)],
            dim=0,
        )
        all_features = self._encode(all_images)

        support_features = [
            x[:B * S].reshape(B, S, x.shape[1], x.shape[2], x.shape[3])
            for x in all_features
        ]
        query_features = [
            x[B * S:].reshape(B * Q, x.shape[1], x.shape[2], x.shape[3])
            for x in all_features
        ]

        f = self.support_proj(support_features[0].reshape(
            B * S, support_features[0].shape[2],
            support_features[0].shape[3], support_features[0].shape[4]
        ))
        f = f.reshape(B, S, -1, f.shape[-2], f.shape[-1])

        prototypes = self.prototype_extractor(
            f, support_masks, self.class_ids, self.k_shot
        )
        task = self.hyper(prototypes)

        # Expand task context for each query.
        context_q = task["context"][:, None].expand(B, Q, -1).reshape(B * Q, -1)
        # gamma/beta are one scalar per feature scale: [B, num_scales].
        # Do not add an extra expansion dimension (which turns scale 4 into
        # a channel vector and breaks broadcasting in DynamicModulation).
        gamma_q = task["gamma"][:, None].expand(B, Q, -1).reshape(B * Q, -1)
        beta_q = task["beta"][:, None].expand(B, Q, -1).reshape(B * Q, -1)
        fusion_q = task["fusion_weights"][:, None].expand(B, Q, -1).reshape(
            B * Q, -1
        )
        adapter_q = task["adapter_gate"][:, None].expand(B, Q, -1).reshape(
            B * Q, -1
        )
        boundary_q = task["boundary_scale"][:, None].expand(B, Q, -1).reshape(
            B * Q, -1
        )
        proto_q = task["prototype_scale"][:, None].expand(B, Q, -1).reshape(
            B * Q, -1
        )

        adapted = []
        for i, x in enumerate(query_features):
            x = self.modulation[i](
                x, context_q, gamma_q[:, i], beta_q[:, i]
            )
            x = self.adapters[i](x, context_q, adapter_q)
            adapted.append(x)

        fused = self.fusion(adapted, context_q, fusion_q)
        proto_flat = prototypes.reshape(B, len(self.class_ids), -1)
        proto_flat = proto_flat[:, None].expand(B, Q, *proto_flat.shape[1:])
        proto_flat = proto_flat.reshape(B * Q, *proto_flat.shape[2:])

        logits = self.decoder(fused, proto_flat, proto_q)
        boundary = self.boundary(fused, context_q, boundary_q)

        logits = F.interpolate(
            logits, size=(Hq, Wq), mode="bilinear", align_corners=False
        )
        boundary = F.interpolate(
            boundary, size=(Hq, Wq), mode="bilinear", align_corners=False
        )

        logits = logits.reshape(B, Q, self.num_classes, Hq, Wq)
        boundary = boundary.reshape(B, Q, 1, Hq, Wq)

        out = {
            "logits": logits[:, 0] if Q == 1 else logits,
            "boundary": boundary[:, 0] if Q == 1 else boundary,
            "prototypes": prototypes,
            "context": task["context"],
            "fusion_weights": task["fusion_weights"],
            "adapter_gate": task["adapter_gate"],
        }
        if return_features:
            out["query_features"] = adapted
            out["fused_features"] = fused
        return out
