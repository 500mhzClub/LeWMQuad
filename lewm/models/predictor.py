"""Transformer predictor with AdaLN action conditioning."""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class AdaLN(nn.Module):
    """Adaptive LayerNorm conditioned on per-step actions."""

    def __init__(self, hidden_dim: int, cond_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.proj = nn.Linear(cond_dim, 2 * hidden_dim)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        scale, shift = self.proj(cond).chunk(2, dim=-1)
        return self.norm(x) * (1.0 + scale) + shift


class TransformerBlock(nn.Module):
    """Pre-norm transformer block with AdaLN."""

    def __init__(
        self,
        hidden_dim: int,
        n_heads: int,
        cond_dim: int,
        mlp_ratio: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.adaln1 = AdaLN(hidden_dim, cond_dim)
        self.attn = nn.MultiheadAttention(
            hidden_dim,
            n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.adaln2 = AdaLN(hidden_dim, cond_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * mlp_ratio),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * mlp_ratio, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        cond: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        h = self.adaln1(x, cond)
        h, _ = self.attn(h, h, h, attn_mask=attn_mask, need_weights=False)
        x = x + h

        h = self.adaln2(x, cond)
        x = x + self.mlp(h)
        return x


class TransformerPredictor(nn.Module):
    """Paper-aligned latent dynamics model.

    The paper uses a ViT-S-sized hidden path: 6 layers, 16 heads, 10% dropout,
    with a 192-D latent projected into a 384-D transformer hidden space.
    """

    def __init__(
        self,
        latent_dim: int = 192,
        hidden_dim: int = 384,
        cmd_dim: int = 3,
        n_layers: int = 6,
        n_heads: int = 16,
        mlp_ratio: int = 4,
        dropout: float = 0.1,
        max_seq_len: int = 4,
    ):
        super().__init__()
        if hidden_dim % n_heads != 0:
            raise ValueError(
                f"hidden_dim={hidden_dim} must be divisible by n_heads={n_heads}"
            )

        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.max_seq_len = max_seq_len

        self.action_embed = nn.Sequential(
            nn.Linear(cmd_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.input_proj = nn.Linear(latent_dim, hidden_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, max_seq_len, hidden_dim))
        self.input_drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    hidden_dim=hidden_dim,
                    n_heads=n_heads,
                    cond_dim=hidden_dim,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                )
                for _ in range(n_layers)
            ]
        )
        self.output_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, latent_dim),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        adaln_projs = {block.adaln1.proj for block in self.blocks}
        adaln_projs.update(block.adaln2.proj for block in self.blocks)
        for module in self.modules():
            if isinstance(module, nn.Linear) and module not in adaln_projs:
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    @staticmethod
    def _causal_mask(steps: int, device: torch.device) -> torch.Tensor:
        return torch.triu(
            torch.ones(steps, steps, device=device, dtype=torch.bool),
            diagonal=1,
        )

    def forward(
        self,
        z_seq: torch.Tensor,
        cmd_seq: torch.Tensor,
    ) -> torch.Tensor:
        batch, steps, _ = z_seq.shape
        if steps > self.max_seq_len:
            raise ValueError(
                f"Sequence length {steps} exceeds max_seq_len={self.max_seq_len}"
            )

        x = self.input_proj(z_seq)
        x = self.input_drop(x + self.pos_embed[:, :steps, :])
        cond = self.action_embed(cmd_seq)
        mask = self._causal_mask(steps, z_seq.device)

        for block in self.blocks:
            x = block(x, cond, attn_mask=mask)

        return self.output_proj(x)

    def rollout(
        self,
        z_start: torch.Tensor,
        action_seq: torch.Tensor,
    ) -> torch.Tensor:
        batch, horizon, _ = action_seq.shape
        _ = batch

        z_buffer = [z_start.unsqueeze(1)]
        preds = []

        for step in range(horizon):
            z_ctx = torch.cat(z_buffer, dim=1)
            a_ctx = action_seq[:, : step + 1, :]
            pred = self.forward(z_ctx, a_ctx)
            z_next = pred[:, -1, :]
            preds.append(z_next)
            z_buffer.append(z_next.unsqueeze(1))

        return torch.stack(preds, dim=1)

    def predict_step(
        self,
        z_history: torch.Tensor,
        cmd_history: torch.Tensor,
    ) -> torch.Tensor:
        return self.forward(z_history, cmd_history)[:, -1, :]
