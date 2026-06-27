from __future__ import annotations

import math
from typing import Dict, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialBranch(nn.Module):
    # Extract spatial feature maps from the RGB input.
    def __init__(
        self,
        backbone_name: str = "xception41",
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        try:
            import timm
        except ImportError as exc:
            raise ImportError("SpatialBranch requires timm. Install with: pip install timm") from exc

        # The backbone returns feature maps because classification is handled separately.
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            num_classes=0,
            global_pool="",
            in_chans=3,
        )
        self.out_channels = int(getattr(self.backbone, "num_features", 2048))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Return a 4D spatial feature map for pooling or fusion.
        features = self.backbone.forward_features(x)
        if features.ndim != 4:
            raise RuntimeError(f"Expected [B, C, H, W], got {tuple(features.shape)}")
        return features


class SpatialBaselineModel(nn.Module):
    # Baseline model that uses only spatial RGB features.
    def __init__(
        self,
        backbone_name: str = "xception41",
        pretrained: bool = True,
        hidden_dim: int = 512,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.spatial_branch = SpatialBranch(
            backbone_name=backbone_name,
            pretrained=pretrained,
        )
        # Global average pooling converts the feature map into one vector per sample.
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Linear(self.spatial_branch.out_channels, hidden_dim)
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, spatial: torch.Tensor, return_debug: bool = False):
        # In debug mode, return intermediate tensors for analysis and visualization.
        # Process the normalized RGB input through the spatial branch.
        spatial_features = self.spatial_branch(spatial)
        hidden = self.proj(self.pool(spatial_features).flatten(1))
        logits = self.classifier(hidden)

        # Debug mode also returns intermediate outputs for inspection.
        if return_debug:
            return logits, {
                "spatial_features": spatial_features,
                "hidden": hidden,
            }
        return logits


class DifferentiableDCT2D(nn.Module):
    # Keep the DCT differentiable so the frequency branch can be trained end to end.
    def __init__(self) -> None:
        super().__init__()
        self._basis_cache: Dict[Tuple[int, str, str], torch.Tensor] = {}

    def _dct_matrix(self, n: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        # Cache basis matrices because many samples share the same spatial sizes.
        key = (n, str(device), str(dtype))
        if key in self._basis_cache:
            return self._basis_cache[key]

        positions = torch.arange(n, device=device, dtype=dtype).unsqueeze(0)
        freqs = torch.arange(n, device=device, dtype=dtype).unsqueeze(1)
        basis = torch.cos((math.pi / n) * (positions + 0.5) * freqs)
        basis[0, :] *= math.sqrt(1.0 / n)
        if n > 1:
            basis[1:, :] *= math.sqrt(2.0 / n)
        self._basis_cache[key] = basis
        return basis

    def dct2(self, x: torch.Tensor) -> torch.Tensor:
        # Apply a 2D DCT using the row and column basis matrices.
        h, w = x.shape
        ch = self._dct_matrix(h, x.device, x.dtype)
        cw = self._dct_matrix(w, x.device, x.dtype)
        return ch @ x @ cw.transpose(0, 1)

    def idct2(self, x: torch.Tensor) -> torch.Tensor:
        # Map the masked spectrum back to the image domain with the inverse transform.
        h, w = x.shape
        ch = self._dct_matrix(h, x.device, x.dtype)
        cw = self._dct_matrix(w, x.device, x.dtype)
        return ch.transpose(0, 1) @ x @ cw


class FrequencyBranch(nn.Module):
    # Learn artifact-focused features from raw crops in the frequency domain.
    def __init__(
        self,
        artifact_size: int = 256,
        base_mask_size: int = 64,
        out_channels: int = 256,
        use_learnable_mask: bool = True,
    ) -> None:
        super().__init__()
        self.artifact_size = int(artifact_size)
        self.dct = DifferentiableDCT2D()

        # Keep the learnable mask enabled by default.
        # Set this to False only for the dual_nomask ablation, which sends the
        # normalized spectrum directly to the inverse DCT.
        self.use_learnable_mask = bool(use_learnable_mask)

        # tanh(x/2) matches the bounded mask formulation used in the methodology.
        # Keep the mask learnable so the model can emphasize informative frequency regions.
        if self.use_learnable_mask:
            self.mask_logits = nn.Parameter(torch.full((1, 1, base_mask_size, base_mask_size), 2.0, dtype=torch.float32))
        else:
            # The no-mask ablation does not create learnable mask parameters.
            self.register_parameter("mask_logits", None)

        mid = max(64, out_channels // 4)
        # Encode the reconstructed artifact map with a small CNN.
        self.encoder = nn.Sequential(
            nn.Conv2d(1, mid, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(mid),
            nn.GELU(),
            nn.Conv2d(mid, mid, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(mid),
            nn.GELU(),
            nn.Conv2d(mid, out_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )
        self.out_channels = int(out_channels)

    @staticmethod
    def _rgb_to_luma(crop: torch.Tensor) -> torch.Tensor:
        # Run frequency analysis on luminance instead of full RGB.
        if crop.shape[0] != 3:
            raise ValueError(f"Expected raw crop with 3 channels, got {tuple(crop.shape)}")
        weights = crop.new_tensor([0.299, 0.587, 0.114]).view(3, 1, 1)
        return (crop * weights).sum(dim=0)

    @staticmethod
    def _normalize_spectrum(spectrum: torch.Tensor) -> torch.Tensor:
        # Apply log scaling to reduce the dominance of large coefficients.
        x = torch.log1p(torch.abs(spectrum))
        mean = x.mean()
        std = x.std().clamp_min(1e-6)
        return (x - mean) / std

    def _bounded_mask(self, h: int, w: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        # Resize the learnable mask to the current crop size and bound it with tanh.
        if self.mask_logits is None:
            raise RuntimeError("mask_logits is None because use_learnable_mask=False")
        mask = F.interpolate(
            self.mask_logits.to(device=device, dtype=dtype),
            size=(h, w),
            mode="bilinear",
            align_corners=False,
            antialias=False,
        )[0, 0]
        return torch.tanh(0.5 * mask)

    def _resize_artifact(self, artifact: torch.Tensor) -> torch.Tensor:
        # Resize artifact maps to a fixed resolution before encoding.
        x = artifact.unsqueeze(0).unsqueeze(0)
        h, w = artifact.shape

        down_h = min(h, self.artifact_size)
        down_w = min(w, self.artifact_size)
        if down_h != h or down_w != w:
            x = F.adaptive_avg_pool2d(x, output_size=(down_h, down_w))

        if down_h != self.artifact_size or down_w != self.artifact_size:
            x = F.interpolate(
                x,
                size=(self.artifact_size, self.artifact_size),
                mode="bilinear",
                align_corners=False,
                antialias=False,
            )
        return x[0, 0]

    def _artifact_from_crop(self, crop: torch.Tensor) -> torch.Tensor:
        # Convert one crop into an artifact map that highlights frequency patterns.
        crop = crop.float()
        luma = self._rgb_to_luma(crop)
        # Center the signal so the transform focuses on variation rather than absolute intensity.
        luma = luma - luma.mean()

        spectrum = self.dct.dct2(luma)
        spectrum = self._normalize_spectrum(spectrum)

        if self.use_learnable_mask:
            h, w = spectrum.shape
            bounded_mask = self._bounded_mask(h=h, w=w, device=spectrum.device, dtype=spectrum.dtype)
            masked_spectrum = spectrum * bounded_mask
        else:
            # In the no-mask ablation, bypass the learnable mask while keeping the rest of the branch unchanged.
            masked_spectrum = spectrum

        artifact = self.dct.idct2(masked_spectrum)
        artifact = self._resize_artifact(artifact)
        return artifact

    def forward(
        self,
        raw_batch: torch.Tensor,
        raw_sizes: Sequence[Tuple[int, int]],
        return_artifact_batch: bool = False,
    ):
        # Reconstruct artifact maps one sample at a time because crops can have different sizes.
        artifacts = []
        for idx, (h, w) in enumerate(raw_sizes):
            crop = raw_batch[idx, :, :h, :w]
            artifacts.append(self._artifact_from_crop(crop))

        artifact_batch = torch.stack(artifacts, dim=0).unsqueeze(1)
        features = self.encoder(artifact_batch)

        if return_artifact_batch:
            return features, artifact_batch
        return features


class CrossAttentionFusion(nn.Module):
    # Use cross-attention so spatial tokens can attend to frequency cues.
    def __init__(
        self,
        spatial_channels: int,
        frequency_channels: int,
        embed_dim: int = 512,
        num_heads: int = 8,
        token_grid_size: int = 14,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        # Pool both branches to the same token grid before fusion.
        self.spatial_pool = nn.AdaptiveAvgPool2d((token_grid_size, token_grid_size))
        self.frequency_pool = nn.AdaptiveAvgPool2d((token_grid_size, token_grid_size))

        self.spatial_proj = nn.Conv2d(spatial_channels, embed_dim, kernel_size=1, bias=False)
        self.frequency_proj = nn.Conv2d(frequency_channels, embed_dim, kernel_size=1, bias=False)

        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
        )

    @staticmethod
    def _to_tokens(feature_map: torch.Tensor) -> torch.Tensor:
        # Flatten feature maps into token sequences for multi-head attention.
        b, c, h, w = feature_map.shape
        return feature_map.view(b, c, h * w).transpose(1, 2).contiguous()

    def forward(self, spatial_features: torch.Tensor, frequency_features: torch.Tensor):
        # Use spatial tokens as queries and frequency tokens as context.
        spatial_tokens = self._to_tokens(self.spatial_proj(self.spatial_pool(spatial_features)))
        frequency_tokens = self._to_tokens(self.frequency_proj(self.frequency_pool(frequency_features)))

        attn_out, _ = self.attn(
            query=spatial_tokens,
            key=frequency_tokens,
            value=frequency_tokens,
            need_weights=False,
        )
        fused_tokens = self.norm(spatial_tokens + self.dropout(attn_out))
        fused_tokens = fused_tokens + self.mlp(fused_tokens)
        fused_vector = fused_tokens.mean(dim=1)
        return fused_vector, fused_tokens


class DualBranchModel(nn.Module):
    # Full model that combines spatial and frequency information.
    def __init__(
        self,
        backbone_name: str = "xception41",
        spatial_pretrained: bool = True,
        fusion_embed_dim: int = 512,
        fusion_heads: int = 8,
        fusion_token_grid_size: int = 14,
        dropout: float = 0.2,
        frequency_artifact_size: int = 256,
        frequency_mask_size: int = 64,
        frequency_out_channels: int = 256,
        use_learnable_mask: bool = True,
    ) -> None:
        super().__init__()
        self.spatial_branch = SpatialBranch(
            backbone_name=backbone_name,
            pretrained=spatial_pretrained,
        )
        self.frequency_branch = FrequencyBranch(
            artifact_size=frequency_artifact_size,
            base_mask_size=frequency_mask_size,
            out_channels=frequency_out_channels,
            # Keep the learnable mask enabled by default and disable it only in dual_nomask.
            use_learnable_mask=use_learnable_mask,
        )
        self.fusion = CrossAttentionFusion(
            spatial_channels=self.spatial_branch.out_channels,
            frequency_channels=self.frequency_branch.out_channels,
            embed_dim=fusion_embed_dim,
            num_heads=fusion_heads,
            token_grid_size=fusion_token_grid_size,
            dropout=dropout,
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(fusion_embed_dim),
            nn.Dropout(dropout),
            nn.Linear(fusion_embed_dim, 2),
        )

    def forward(
        self,
        spatial: torch.Tensor,
        raw: torch.Tensor,
        raw_sizes: Sequence[Tuple[int, int]],
        return_debug: bool = False,
    ):
        spatial_features = self.spatial_branch(spatial)

        if return_debug:
            frequency_features, artifact_batch = self.frequency_branch(
                raw_batch=raw,
                raw_sizes=raw_sizes,
                return_artifact_batch=True,
            )
        else:
            frequency_features = self.frequency_branch(
                raw_batch=raw,
                raw_sizes=raw_sizes,
                return_artifact_batch=False,
            )
            artifact_batch = None

        # Fusion produces the final representation for binary classification.
        fused_vector, fused_tokens = self.fusion(spatial_features, frequency_features)
        logits = self.classifier(fused_vector)

        if return_debug:
            return logits, {
                "spatial_features": spatial_features,
                "frequency_features": frequency_features,
                "artifact_batch": artifact_batch,
                "fused_tokens": fused_tokens,
            }
        return logits


def build_model(model_type: str, **kwargs) -> nn.Module:
    # Centralize model creation for both training and evaluation.
    model_type = model_type.lower().strip()
    if model_type == "spatial":
        return SpatialBaselineModel(
            backbone_name=kwargs.get("backbone_name", "xception41"),
            pretrained=kwargs.get("pretrained", True),
            hidden_dim=kwargs.get("hidden_dim", 512),
            dropout=kwargs.get("dropout", 0.2),
        )
    if model_type == "dual":
        return DualBranchModel(
            backbone_name=kwargs.get("backbone_name", "xception41"),
            spatial_pretrained=kwargs.get("pretrained", True),
            fusion_embed_dim=kwargs.get("fusion_embed_dim", 512),
            fusion_heads=kwargs.get("fusion_heads", 8),
            fusion_token_grid_size=kwargs.get("fusion_token_grid_size", 14),
            dropout=kwargs.get("dropout", 0.2),
            frequency_artifact_size=kwargs.get("frequency_artifact_size", 256),
            frequency_mask_size=kwargs.get("frequency_mask_size", 64),
            frequency_out_channels=kwargs.get("frequency_out_channels", 256),
            # Preserve the original dual-model behavior.
            use_learnable_mask=True,
        )
    if model_type == "dual_nomask":
        # Use the same dual architecture with the learnable mask disabled to measure its contribution.
        return DualBranchModel(
            backbone_name=kwargs.get("backbone_name", "xception41"),
            spatial_pretrained=kwargs.get("pretrained", True),
            fusion_embed_dim=kwargs.get("fusion_embed_dim", 512),
            fusion_heads=kwargs.get("fusion_heads", 8),
            fusion_token_grid_size=kwargs.get("fusion_token_grid_size", 14),
            dropout=kwargs.get("dropout", 0.2),
            frequency_artifact_size=kwargs.get("frequency_artifact_size", 256),
            frequency_mask_size=kwargs.get("frequency_mask_size", 64),
            frequency_out_channels=kwargs.get("frequency_out_channels", 256),
            use_learnable_mask=False,
        )
    raise ValueError(f"Unsupported model_type: {model_type}. Expected 'spatial', 'dual', or 'dual_nomask'.")


__all__ = [
    "SpatialBranch",
    "FrequencyBranch",
    "CrossAttentionFusion",
    "SpatialBaselineModel",
    "DualBranchModel",
    "build_model",
]
