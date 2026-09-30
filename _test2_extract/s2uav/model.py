"""SegFormer-style multi-scale segmentation with scene-conditioned adapters."""
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden=None):
        super().__init__()
        hidden = hidden or max(in_dim, out_dim)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.GELU(), nn.Linear(hidden, out_dim)
        )

    def forward(self, x):
        return self.net(x)


class MixBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=2):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.norm = nn.BatchNorm2d(out_channels)
        self.mlp = nn.Sequential(
            nn.Conv2d(out_channels, out_channels * 4, 1), nn.GELU(),
            nn.Conv2d(out_channels * 4, out_channels, 1),
        )
        self.skip = (
            nn.Conv2d(in_channels, out_channels, 1, stride=stride)
            if in_channels != out_channels or stride != 1
            else nn.Identity()
        )

    def forward(self, x):
        y = F.gelu(self.norm(self.proj(x)))
        return y + self.mlp(y) + self.skip(x)


class SceneCondition(nn.Module):
    def __init__(self, channels, dim=128):
        super().__init__()
        self.projections = nn.ModuleList([nn.Conv2d(c, dim, 1) for c in channels])
        self.mlp = MLP(dim * len(channels), dim, dim)

    def forward(self, features):
        pooled = [
            F.adaptive_avg_pool2d(projection(feature), 1).flatten(1)
            for projection, feature in zip(self.projections, features)
        ]
        return self.mlp(torch.cat(pooled, dim=1))


class DynamicModulation(nn.Module):
    def __init__(self, context_dim, channels, scale=0.15):
        super().__init__()
        self.scale = scale
        self.layers = nn.ModuleList([nn.Linear(context_dim, 2 * c) for c in channels])
        self.norms = nn.ModuleList([nn.GroupNorm(1, c) for c in channels])
        for layer in self.layers:
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, features, context):
        result = []
        for feature, layer, norm in zip(features, self.layers, self.norms):
            gamma, beta = layer(context).chunk(2, dim=1)
            gamma = self.scale * torch.tanh(gamma)[:, :, None, None]
            beta = self.scale * torch.tanh(beta)[:, :, None, None]
            result.append(norm(feature) * (1 + gamma) + beta)
        return result


class DynamicLowRankAdapter(nn.Module):
    """Generate a small per-sample low-rank residual for fused features."""
    def __init__(self, context_dim, channels, rank=8, scale=0.1):
        super().__init__()
        self.channels, self.rank, self.scale = channels, rank, scale
        self.generator = nn.Linear(context_dim, 2 * channels * rank)
        nn.init.zeros_(self.generator.weight)
        nn.init.zeros_(self.generator.bias)

    def forward(self, feature, context):
        params = self.generator(context)
        size = self.channels * self.rank
        down = params[:, :size].reshape(-1, self.rank, self.channels)
        up = params[:, size:].reshape(-1, self.channels, self.rank)
        latent = torch.einsum("brc,bchw->brhw", down, feature)
        residual = torch.einsum("bcr,brhw->bchw", up, latent)
        return feature + self.scale * residual / max(1, self.rank)


class DynamicFusion(nn.Module):
    def __init__(self, channels, context_dim, out_channels):
        super().__init__()
        self.weights = MLP(context_dim, len(channels), context_dim)
        self.projections = nn.ModuleList([nn.Conv2d(c, out_channels, 1) for c in channels])

    def forward(self, features, context, output_size):
        weights = self.weights(context).softmax(dim=1)
        fused = 0
        for index, (feature, projection) in enumerate(zip(features, self.projections)):
            x = projection(feature)
            x = F.interpolate(x, output_size, mode="bilinear", align_corners=False)
            fused = fused + weights[:, index, None, None, None] * x
        return fused


class SegFormerB3Encoder(nn.Module):
    """Pretrained Hugging Face SegFormer-B3 feature pyramid."""
    def __init__(self, pretrained=True, model_name="nvidia/mit-b3"):
        super().__init__()
        try:
            from transformers import SegformerModel
        except ImportError as error:
            raise ImportError("Install transformers to use the SegFormer-B3 encoder") from error
        local_model = Path(__file__).resolve().parents[1] / "models" / model_name.replace("/", "--")
        model_source = str(local_model) if local_model.is_dir() else model_name
        self.backbone = (
            SegformerModel.from_pretrained(model_source)
            if pretrained
            else SegformerModel.from_pretrained(model_source, local_files_only=True)
        )
        self.channels = [64, 128, 320, 512]

    def forward(self, image):
        output = self.backbone(pixel_values=image, output_hidden_states=True)
        # SegFormer hidden states are channel-last: B x HW x C.
        features = []
        for hidden, channel in zip(output.hidden_states[-4:], self.channels):
            if hidden.ndim == 4:
                features.append(hidden)
                continue
            batch, tokens, hidden_channels = hidden.shape
            height = width = int(tokens ** 0.5)
            if height * width != tokens:
                raise RuntimeError(f"Unexpected SegFormer token shape: {hidden.shape}")
            features.append(hidden.transpose(1, 2).reshape(batch, hidden_channels, height, width))
        return features


class HyperSegUAV(nn.Module):
    """HyperSeg-UAV with a pretrained SegFormer-B3 encoder."""
    def __init__(self, classes=9, widths=(64, 128, 320, 512), context_dim=128,
                 pretrained=True, model_name="nvidia/mit-b3", adapter_rank=8,
                 adapter_scale=0.1):
        super().__init__()
        self.model_config = {
            "classes": classes,
            "widths": list(widths),
            "context_dim": context_dim,
            "pretrained": False,
            "model_name": model_name,
            "adapter_rank": adapter_rank,
            "adapter_scale": adapter_scale,
        }
        self.encoder = SegFormerB3Encoder(pretrained=pretrained, model_name=model_name)
        self.scene = SceneCondition(widths, context_dim)
        self.modulation = DynamicModulation(context_dim, widths)
        self.fusion = DynamicFusion(widths, context_dim, widths[0])
        self.low_rank = DynamicLowRankAdapter(context_dim, widths[0], adapter_rank, adapter_scale)
        self.decoder = nn.Sequential(
            nn.Conv2d(widths[0], widths[0], 3, padding=1, bias=False),
            nn.BatchNorm2d(widths[0]),
            nn.GELU(),
            nn.Conv2d(widths[0], widths[0], 3, padding=1, bias=False),
            nn.BatchNorm2d(widths[0]), nn.GELU(),
        )
        self.head = nn.Conv2d(widths[0], classes, 1)
        self.boundary = nn.Conv2d(widths[0], 1, 1)

    def forward(self, image):
        input_size = image.shape[-2:]
        features = self.encoder(image)
        context = self.scene(features)
        features = self.modulation(features, context)
        fused = self.fusion(features, context, features[0].shape[-2:])
        fused = self.low_rank(fused, context)
        decoded = self.decoder(fused)
        logits = F.interpolate(
            self.head(decoded), input_size, mode="bilinear", align_corners=False
        )
        boundary = F.interpolate(
            self.boundary(decoded), input_size, mode="bilinear", align_corners=False
        )
        return {"logits": logits, "boundary": boundary, "context": context}
