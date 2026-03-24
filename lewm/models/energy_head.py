"""Learned scalar energy head for latent-space planning."""
from __future__ import annotations

import torch
import torch.nn as nn


class GoalEnergyHead(nn.Module):
    """Scores compatibility between a predicted latent and a goal latent.

    Input: concatenation of [z_pred, z_goal, z_pred - z_goal, z_pred * z_goal].
    Output: scalar energy (lower = more compatible).
    """

    def __init__(self, latent_dim: int = 192, dropout: float = 0.0):
        super().__init__()
        in_dim = latent_dim * 4
        self.net = nn.Sequential(
            nn.Linear(in_dim, 1024), nn.LayerNorm(1024), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(1024, 512), nn.LayerNorm(512), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(512, 1),
        )

    def forward(self, z_pred: torch.Tensor, z_goal: torch.Tensor) -> torch.Tensor:
        x = torch.cat([z_pred, z_goal, z_pred - z_goal, z_pred * z_goal], dim=-1)
        return self.net(x).squeeze(-1)
