"""Transformer predictor with Adaptive Layer Normalization (AdaLN).

Replaces the GRU-based LatentPredictor from TinyQuadJEPA-v2.

Key design choices (following the LeWM paper):
  - Actions condition every layer via AdaLN (zero-initialised so action
    influence grows gradually during training).
  - Causal (temporal) masking prevents attending to future embeddings.
  - Learned positional embeddings encode temporal order.
  - A separate Projector (BatchNorm) maps outputs to the comparison space.

During training the predictor receives teacher-forced encoder outputs.
During planning it is unrolled auto-regressively over a growing buffer.
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# Adaptive Layer Normalisation (AdaLN)
# --------------------------------------------------------------------------- #

class AdaLN(nn.Module):
    """Adaptive LayerNorm conditioned on a per-position signal.

    Produces per-position scale (γ) and shift (β) from the conditioning
    vector.  Weights are zero-initialised so the initial behaviour is plain
    LayerNorm, and action influence ramps up during training.
    """

    def __init__(self, hidden_dim: int, cond_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.proj = nn.Linear(cond_dim, 2 * hidden_dim)
        # Zero-init for progressive conditioning
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:    (B, T, D) — input features.
            cond: (B, T, D_cond) — per-position conditioning.
        Returns:
            (B, T, D) — normalised + modulated features.
        """
        scale, shift = self.proj(cond).chunk(2, dim=-1)
        return self.norm(x) * (1.0 + scale) + shift


# --------------------------------------------------------------------------- #
# Transformer block with AdaLN
# --------------------------------------------------------------------------- #

class TransformerBlock(nn.Module):
    """Pre-norm transformer block with AdaLN for action conditioning."""

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
            hidden_dim, n_heads, dropout=dropout, batch_first=True,
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
        """
        Args:
            x:         (B, T, D)
            cond:      (B, T, D_cond)
            attn_mask: (T, T) causal mask (True = masked).
        """
        # Self-attention with AdaLN pre-norm
        h = self.adaln1(x, cond)
        h, _ = self.attn(h, h, h, attn_mask=attn_mask, need_weights=False)
        x = x + h

        # MLP with AdaLN pre-norm
        h = self.adaln2(x, cond)
        x = x + self.mlp(h)
        return x


# --------------------------------------------------------------------------- #
# Full Transformer Predictor
# --------------------------------------------------------------------------- #

class TransformerPredictor(nn.Module):
    """Action-conditioned transformer dynamics model.

    Given a sequence of encoder embeddings z_{1:T} and actions a_{1:T},
    predicts z_{2:T+1} (the next embedding at every position).

    Architecture sized for the quadruped task (~3 M params):
        4 layers, 8 heads, 256 hidden dim, 10 % dropout.
    """

    def __init__(
        self,
        latent_dim: int = 256,
        cmd_dim: int = 3,
        n_layers: int = 4,
        n_heads: int = 8,
        mlp_ratio: int = 4,
        dropout: float = 0.1,
        max_seq_len: int = 64,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.max_seq_len = max_seq_len

        # Action embedding
        self.action_embed = nn.Sequential(
            nn.Linear(cmd_dim, latent_dim),
            nn.SiLU(),
            nn.Linear(latent_dim, latent_dim),
        )

        # Learned positional embeddings
        self.pos_embed = nn.Parameter(torch.zeros(1, max_seq_len, latent_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # Input projection (latent → hidden)
        self.input_proj = nn.Linear(latent_dim, latent_dim)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                hidden_dim=latent_dim,
                n_heads=n_heads,
                cond_dim=latent_dim,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
            )
            for _ in range(n_layers)
        ])

        # Output projection
        self.output_proj = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, latent_dim),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear) and m not in [
                b.adaln1.proj for b in self.blocks
            ] + [b.adaln2.proj for b in self.blocks]:
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    @staticmethod
    def _causal_mask(T: int, device: torch.device) -> torch.Tensor:
        """Upper-triangular causal mask (True = blocked)."""
        return torch.triu(
            torch.ones(T, T, device=device, dtype=torch.bool), diagonal=1,
        )

    def forward(
        self,
        z_seq: torch.Tensor,
        cmd_seq: torch.Tensor,
    ) -> torch.Tensor:
        """Full-sequence forward (training with teacher-forcing).

        Args:
            z_seq:   (B, T, D) — encoder backbone outputs (raw, not projected).
            cmd_seq: (B, T, cmd_dim) — action commands at each step.

        Returns:
            (B, T, D) — predicted *raw* next-embeddings for each position.
            Position t contains the prediction for z_{t+1}.
        """
        B, T, D = z_seq.shape

        x = self.input_proj(z_seq) + self.pos_embed[:, :T, :]
        cond = self.action_embed(cmd_seq)                    # (B, T, D)
        mask = self._causal_mask(T, z_seq.device)            # (T, T)

        for block in self.blocks:
            x = block(x, cond, attn_mask=mask)

        return self.output_proj(x)

    # ------------------------------------------------------------------ #
    # Auto-regressive rollout  (for planning / CEM)
    # ------------------------------------------------------------------ #

    def rollout(
        self,
        z_start: torch.Tensor,
        action_seq: torch.Tensor,
    ) -> torch.Tensor:
        """Auto-regressive rollout for latent planning.

        Args:
            z_start:    (B, D) — initial encoder embedding (raw).
            action_seq: (B, H, cmd_dim) — planned action sequence.

        Returns:
            (B, H, D) — predicted raw latent at each horizon step.
        """
        B, H, _ = action_seq.shape
        device = z_start.device

        z_buffer = [z_start.unsqueeze(1)]          # list of (B, 1, D)
        preds = []

        for t in range(H):
            # Current context: z_1, z_2, …, z_{t+1}
            z_ctx = torch.cat(z_buffer, dim=1)     # (B, t+1, D)
            a_ctx = action_seq[:, : t + 1, :]      # (B, t+1, cmd_dim)

            pred = self.forward(z_ctx, a_ctx)       # (B, t+1, D)
            z_next = pred[:, -1, :]                 # (B, D) — last position

            preds.append(z_next)
            z_buffer.append(z_next.unsqueeze(1))

        return torch.stack(preds, dim=1)            # (B, H, D)

    def predict_step(
        self,
        z_history: torch.Tensor,
        cmd_history: torch.Tensor,
    ) -> torch.Tensor:
        """Single-step prediction given a context window.

        Convenience wrapper for code that previously used the GRU step API.

        Args:
            z_history:   (B, N, D) — recent latent history.
            cmd_history: (B, N, cmd_dim) — corresponding actions.

        Returns:
            (B, D) — predicted next latent.
        """
        return self.forward(z_history, cmd_history)[:, -1, :]
