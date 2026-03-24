"""LeWorldModel — stable end-to-end JEPA from pixels.

Replaces the EMA student-teacher CanonicalJEPA with the LeWM approach:
  - Single encoder (no target encoder, no EMA, no stop-gradient).
  - Transformer predictor with AdaLN action conditioning.
  - Two-term loss: MSE prediction + λ·SIGReg anti-collapse.
  - BatchNorm projectors map encoder/predictor outputs to the space where
    the loss is computed.

Reference:
    Maes, Le Lidec, Scieur, LeCun, Balestriero.
    "LeWorldModel: Stable End-to-End Joint-Embedding Predictive Architecture
     from Pixels", arXiv:2603.19312, Mar 2026.
"""
from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoders import JointEncoder, Projector
from .predictor import TransformerPredictor
from .sigreg import sigreg_stepwise


class LeWorldModel(nn.Module):
    """End-to-end latent world model with SIGReg regularisation.

    Components
    ----------
    encoder : JointEncoder
        Maps (vision, proprio) → z_raw  (backbone, ends with LayerNorm).
    enc_projector : Projector
        z_raw → z_proj  (1-layer MLP + BatchNorm, comparison space).
    predictor : TransformerPredictor
        (z_raw_{1:T}, cmd_{1:T}) → z_pred_raw_{1:T}  (next-step predictions).
    pred_projector : Projector
        z_pred_raw → z_pred_proj  (same architecture, separate weights).
    """

    def __init__(
        self,
        latent_dim: int = 256,
        cmd_dim: int = 3,
        pred_layers: int = 4,
        pred_heads: int = 8,
        pred_mlp_ratio: int = 4,
        pred_dropout: float = 0.1,
        max_seq_len: int = 64,
        sigreg_lambda: float = 0.1,
        sigreg_projections: int = 512,
        sigreg_knots: int = 17,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.sigreg_lambda = sigreg_lambda
        self.sigreg_projections = sigreg_projections
        self.sigreg_knots = sigreg_knots

        # --- Encoder (single, gets gradients) ---
        self.encoder = JointEncoder(latent_dim=latent_dim)
        self.enc_projector = Projector(latent_dim, latent_dim)

        # --- Predictor ---
        self.predictor = TransformerPredictor(
            latent_dim=latent_dim,
            cmd_dim=cmd_dim,
            n_layers=pred_layers,
            n_heads=pred_heads,
            mlp_ratio=pred_mlp_ratio,
            dropout=pred_dropout,
            max_seq_len=max_seq_len,
        )
        self.pred_projector = Projector(latent_dim, latent_dim)

    # ------------------------------------------------------------------ #
    # Encoding helpers
    # ------------------------------------------------------------------ #

    def encode(
        self, vis: torch.Tensor, prop: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode a single frame.

        Returns:
            z_raw:  (B, D) backbone output  (for predictor input / rollout).
            z_proj: (B, D) projected output (for loss / goal matching).
        """
        z_raw = self.encoder(vis, prop)
        z_proj = self.enc_projector(z_raw)
        return z_raw, z_proj

    def encode_seq(
        self,
        vis_seq: torch.Tensor,
        prop_seq: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode a temporal sequence of frames.

        Args:
            vis_seq:  (B, T, C, H, W)
            prop_seq: (B, T, P)

        Returns:
            z_raw:  (B, T, D)
            z_proj: (B, T, D)
        """
        B, T = vis_seq.shape[:2]
        vis_flat = vis_seq.reshape(B * T, *vis_seq.shape[2:])
        prop_flat = prop_seq.reshape(B * T, *prop_seq.shape[2:])

        z_raw_flat = self.encoder(vis_flat, prop_flat)         # (B*T, D)
        z_proj_flat = self.enc_projector(z_raw_flat)           # (B*T, D)

        z_raw = z_raw_flat.reshape(B, T, -1)
        z_proj = z_proj_flat.reshape(B, T, -1)
        return z_raw, z_proj

    # ------------------------------------------------------------------ #
    # Training forward
    # ------------------------------------------------------------------ #

    def forward(
        self,
        vis_seq: torch.Tensor,
        prop_seq: torch.Tensor,
        cmd_seq: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor]:
        """Full training forward pass.

        Args:
            vis_seq:  (B, T, 3, 64, 64) — uint8→float already handled.
            prop_seq: (B, T, 47)
            cmd_seq:  (B, T, 3)
            mask:     (B, T-1) bool — True for valid transitions.

        Returns:
            dict with keys:
                loss       — total loss  (pred + λ·sigreg).
                pred_loss  — MSE prediction loss.
                sigreg_loss — SIGReg regularisation.
                z_proj_std — per-dim std of projected embeddings (collapse check).
        """
        B, T = vis_seq.shape[:2]

        # 1. Encode full sequence
        z_raw, z_proj = self.encode_seq(vis_seq, prop_seq)   # (B, T, D) each

        # 2. Predict next embeddings (teacher-forcing)
        z_pred_raw = self.predictor(z_raw, cmd_seq)          # (B, T, D)
        z_pred_proj = self.pred_projector.forward_seq(z_pred_raw)  # (B, T, D)

        # 3. Prediction loss:  z_pred_proj[:, t] should match z_proj[:, t+1]
        pred = z_pred_proj[:, :-1]   # (B, T-1, D) — predictions for steps 1..T-1
        target = z_proj[:, 1:]       # (B, T-1, D) — actual embeddings at 1..T-1

        per_sample_mse = (pred - target).square().mean(dim=-1)  # (B, T-1)

        if mask is not None:
            n_valid = mask.float().sum().clamp(min=1.0)
            pred_loss = (per_sample_mse * mask.float()).sum() / n_valid
        else:
            pred_loss = per_sample_mse.mean()

        # 4. Step-wise SIGReg on projected encoder embeddings
        sig_loss = sigreg_stepwise(
            z_proj,
            n_projections=self.sigreg_projections,
            n_knots=self.sigreg_knots,
        )

        # 5. Total loss
        total_loss = pred_loss + self.sigreg_lambda * sig_loss

        # 6. Collapse monitoring
        z_proj_std = z_proj.detach().float().std(dim=(0, 1)).mean()

        return {
            "loss": total_loss,
            "pred_loss": pred_loss,
            "sigreg_loss": sig_loss,
            "z_proj_std": z_proj_std,
        }

    # ------------------------------------------------------------------ #
    # Planning helpers  (used by CEM / MPC)
    # ------------------------------------------------------------------ #

    def plan_rollout(
        self,
        z_start_raw: torch.Tensor,
        action_seq: torch.Tensor,
    ) -> torch.Tensor:
        """Rollout predictor auto-regressively for latent planning.

        Args:
            z_start_raw: (B, D) — raw encoder embedding of the current frame.
            action_seq:  (B, H, cmd_dim) — candidate action sequence.

        Returns:
            z_pred_proj: (B, H, D) — projected predicted latents at each step.
        """
        z_pred_raw = self.predictor.rollout(z_start_raw, action_seq)  # (B, H, D)
        z_pred_proj = self.pred_projector.forward_seq(z_pred_raw)
        return z_pred_proj

    def plan_cost(
        self,
        z_pred_proj: torch.Tensor,
        z_goal_proj: torch.Tensor,
    ) -> torch.Tensor:
        """Terminal goal-matching cost in projected space.

        Args:
            z_pred_proj: (B, H, D) or (B, D) — predicted latents.
            z_goal_proj: (B, D) — goal embedding (projected).

        Returns:
            (B,) — L2² cost.
        """
        if z_pred_proj.dim() == 3:
            z_pred_proj = z_pred_proj[:, -1, :]        # terminal state
        return (z_pred_proj - z_goal_proj).square().sum(dim=-1)

    # ------------------------------------------------------------------ #
    # Backward-compatible convenience methods
    # ------------------------------------------------------------------ #

    def encode_observation(
        self, vis: torch.Tensor, prop: torch.Tensor,
    ) -> torch.Tensor:
        """Return projected embedding (for goal matching / energy head).

        Drop-in replacement for ``model.encode_target(vis, prop)`` in the
        old CanonicalJEPA API.
        """
        _, z_proj = self.encode(vis, prop)
        return z_proj

    def encode_raw(
        self, vis: torch.Tensor, prop: torch.Tensor,
    ) -> torch.Tensor:
        """Return raw backbone embedding (for predictor input)."""
        return self.encoder(vis, prop)
