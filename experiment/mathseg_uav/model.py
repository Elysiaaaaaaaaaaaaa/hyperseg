"""MiT-B3 with controlled mathematical-prior fusion ablations."""

from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


VARIANTS = ("M0", "M1", "M2", "M3", "M4")
PRIOR_NAMES = ("gradient", "variance3", "variance7", "laplacian")


class SpatialPrior(nn.Module):
    """Fixed grayscale operators on augmented RGB in [0, 1], computed in FP32."""

    def __init__(self, channels=PRIOR_NAMES, mode="normal"):
        super().__init__()
        if not channels or set(channels) - set(PRIOR_NAMES):
            raise ValueError(f"prior channels must be a nonempty subset of {PRIOR_NAMES}")
        if len(set(channels)) != len(channels) or mode not in ("normal", "zero", "shuffle"):
            raise ValueError("Duplicate prior channels or invalid prior mode")
        self.channels, self.mode = tuple(channels), mode
        self.register_buffer("gray", torch.tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1))
        self.register_buffer("kernels", torch.tensor([
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
            [[0, 1, 0], [1, -4, 1], [0, 1, 0]],
        ], dtype=torch.float32)[:, None])

    def forward(self, image, size):
        with torch.autocast(device_type=image.device.type, enabled=False):
            gray = (image.float() * self.gray).sum(1, keepdim=True)
            derivatives = F.conv2d(F.pad(gray, (1, 1, 1, 1), mode="replicate"), self.kernels)
            values = {
                "gradient": torch.linalg.vector_norm(derivatives[:, :2], dim=1, keepdim=True),
                "laplacian": derivatives[:, 2:3].abs(),
            }
            for window in (3, 7):
                pad = window // 2
                padded = F.pad(gray, (pad, pad, pad, pad), mode="replicate")
                mean = F.avg_pool2d(padded, window, stride=1)
                values[f"variance{window}"] = (
                    F.avg_pool2d(padded.square(), window, stride=1) - mean.square()
                ).clamp_min(0)
            prior = torch.cat([values[name] for name in self.channels], dim=1)
            # Do not amplify FP32 cancellation noise in constant/flat regions.
            prior = torch.where(prior <= 1e-7, torch.zeros_like(prior), prior)
            # Smooth bounded normalization, independently for each image/channel.
            prior = prior / (prior.mean((2, 3), keepdim=True) + prior + 1e-6)
            prior = F.interpolate(prior, size=size, mode="bilinear", align_corners=False)
            if self.mode == "zero":
                prior = torch.zeros_like(prior)
            elif self.mode == "shuffle":
                # Fixed permutation: reproducible evaluation, no training RNG consumption.
                generator = torch.Generator().manual_seed(1729)
                indices = torch.randperm(prior.shape[-2] * prior.shape[-1], generator=generator)
                prior = prior.flatten(2)[:, :, indices.to(prior.device)].reshape_as(prior)
            return prior


class FusionHead(nn.Module):
    def __init__(self, channels, width, variant, prior_channels):
        super().__init__()
        self.variant = variant
        self.projections = nn.ModuleList([nn.Conv2d(c, width, 1) for c in channels])
        # Initialize common modules before variant-specific modules for paired seeds.
        self.decoder = nn.Sequential(
            nn.Conv2d(width, width, 3, padding=1, bias=False), nn.GroupNorm(1, width), nn.GELU(),
            nn.Conv2d(width, width, 3, padding=1, bias=False), nn.GroupNorm(1, width), nn.GELU(),
        )
        self.segmentation = nn.Conv2d(width, 9, 1)
        self.boundary = nn.Conv2d(width, 1, 1)
        if variant == "M1":
            self.modulation = nn.Conv2d(prior_channels, width, 1)
            nn.init.zeros_(self.modulation.weight)
            nn.init.zeros_(self.modulation.bias)
        if variant in ("M2", "M3", "M4"):
            self.gate = nn.Sequential(
                nn.Conv2d(4 * width + prior_channels, width, 1), nn.GELU(),
                nn.Conv2d(width, 4, 1),
            )
            nn.init.zeros_(self.gate[-1].weight)
            nn.init.zeros_(self.gate[-1].bias)

    def forward(self, features, prior, output_size, modulation_scale=None):
        size = features[0].shape[-2:]
        aligned = [F.interpolate(project(f), size, mode="bilinear", align_corners=False)
                   for project, f in zip(self.projections, features)]
        if self.variant in ("M2", "M3", "M4"):
            gate_prior = torch.zeros_like(prior) if self.variant == "M2" else prior
            weights = self.gate(torch.cat([*aligned, gate_prior.to(aligned[0].dtype)], 1)).softmax(1)
        else:
            weights = aligned[0].new_full((aligned[0].shape[0], 4, *size), 0.25)
        fused = sum(f * weights[:, k:k + 1] for k, f in enumerate(aligned))
        if self.variant == "M1":
            scale = 0.1 if modulation_scale is None else float(modulation_scale)
            fused = fused * (1 + scale * torch.tanh(self.modulation(prior.to(fused.dtype))))
        decoded = self.decoder(fused)
        return {
            "logits": F.interpolate(self.segmentation(decoded), output_size, mode="bilinear", align_corners=False),
            "boundary": F.interpolate(self.boundary(decoded), output_size, mode="bilinear", align_corners=False),
            "scale_weights": weights,
            "prior": prior,
        }


class MathSegUAV(nn.Module):
    def __init__(self, variant="M0", width=64, model_name="nvidia/mit-b3", pretrained=True,
                 local_files_only=True, encoder_config=None, prior_channels=PRIOR_NAMES,
                 prior_mode="normal"):
        super().__init__()
        from transformers import SegformerConfig, SegformerModel

        if variant not in VARIANTS or width < 1:
            raise ValueError("Invalid variant or decoder width")
        local = Path(__file__).resolve().parents[2] / "models" / model_name.replace("/", "--")
        source = str(local) if local.is_dir() else model_name
        if encoder_config is not None:
            if pretrained:
                raise ValueError("Embedded encoder_config must be loaded with pretrained=False")
            self.encoder = SegformerModel(SegformerConfig.from_dict(encoder_config))
        elif pretrained:
            self.encoder = SegformerModel.from_pretrained(source, local_files_only=local_files_only)
        else:
            # An actual random initialization; config still comes from the selected backbone.
            config = SegformerConfig.from_pretrained(source, local_files_only=local_files_only)
            self.encoder = SegformerModel(config)
        channels = self.encoder.config.hidden_sizes
        if len(channels) != 4 or not self.encoder.config.reshape_last_stage:
            raise ValueError("MathSeg requires four spatial feature maps (reshape_last_stage=True)")
        self.prior = SpatialPrior(prior_channels, prior_mode)
        self.head = FusionHead(channels, width, variant, len(prior_channels))
        self.variant = variant
        self.register_buffer("rgb_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("rgb_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self.model_config = dict(
            variant=variant, width=width, model_name=model_name, pretrained=False,
            local_files_only=True, encoder_config=self.encoder.config.to_dict(),
            prior_channels=list(prior_channels), prior_mode=prior_mode,
        )

    def encode_features(self, image):
        features = self.encoder(
            pixel_values=(image - self.rgb_mean) / self.rgb_std,
            output_hidden_states=True, return_dict=True,
        ).hidden_states
        size = features[0].shape[-2:]
        # M2 computes then zeros priors so its operator overhead matches M3.
        prior = (image.new_zeros((image.shape[0], len(self.prior.channels), *size))
                 if self.variant == "M0" else self.prior(image, size))
        return features, prior

    def forward(self, image, modulation_scale=None):
        features, prior = self.encode_features(image)
        return self.head(features, prior, image.shape[-2:], modulation_scale=modulation_scale)
