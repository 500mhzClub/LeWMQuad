"""Vision, proprioception, joint encoders, and BatchNorm projector.

Key change from TinyQuadJEPA-v2:
  - The JointEncoder backbone retains LayerNorm for stable feature extraction.
  - A separate **Projector** with BatchNorm maps backbone output into the
    "comparison space" where SIGReg and the prediction MSE operate.
  - BatchNorm (not LayerNorm) is required because SIGReg tests the *batch*
    distribution against N(0, I); LayerNorm normalises per-sample and would
    mask departures from Gaussianity.
"""
from __future__ import annotations

import torch
import torch.nn as nn


# --------------------------------------------------------------------------- #
# Backbone sub-encoders  (identical to TinyQuadJEPA-v2)
# --------------------------------------------------------------------------- #

class VisionEncoder(nn.Module):
    """4-layer CNN: 64×64 RGB → feature_dim vector."""

    def __init__(self, feature_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1),  nn.ELU(),
            nn.Conv2d(32, 64, 4, 2, 1), nn.ELU(),
            nn.Conv2d(64, 128, 4, 2, 1), nn.ELU(),
            nn.Conv2d(128, 256, 4, 2, 1), nn.ELU(),
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, feature_dim),
            nn.LayerNorm(feature_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ProprioEncoder(nn.Module):
    """MLP: proprio_dim → feature_dim."""

    def __init__(self, input_dim: int = 47, feature_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256), nn.ELU(),
            nn.Linear(256, feature_dim),
            nn.LayerNorm(feature_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# --------------------------------------------------------------------------- #
# Joint encoder  (backbone — ends with LayerNorm)
# --------------------------------------------------------------------------- #

class JointEncoder(nn.Module):
    """Fuses vision + proprio into a single latent vector (backbone output)."""

    def __init__(self, latent_dim: int = 256):
        super().__init__()
        self.vis_enc = VisionEncoder(128)
        self.prop_enc = ProprioEncoder(47, 128)
        self.fusion = nn.Sequential(
            nn.Linear(256, 256), nn.ELU(),
            nn.Linear(256, latent_dim),
            nn.LayerNorm(latent_dim),
        )

    def forward(self, vision: torch.Tensor, proprio: torch.Tensor) -> torch.Tensor:
        z = torch.cat([self.vis_enc(vision), self.prop_enc(proprio)], dim=-1)
        return self.fusion(z)


# --------------------------------------------------------------------------- #
# Projector  (new — maps backbone → comparison space with BatchNorm)
# --------------------------------------------------------------------------- #

class Projector(nn.Module):
    """1-layer MLP with BatchNorm (no affine) for SIGReg compatibility.

    The paper notes that the ViT's final LayerNorm prevents SIGReg from
    working; a BN projector fixes this.  Both the encoder and the predictor
    share the same projector *architecture* (but separate weights).
    """

    def __init__(self, in_dim: int = 256, out_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.BatchNorm1d(out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, D) or (B*T, D)."""
        return self.net(x)

    def forward_seq(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, D) — reshapes for BN then restores shape."""
        B, T, D = x.shape
        return self.net(x.reshape(B * T, -1)).reshape(B, T, -1)
