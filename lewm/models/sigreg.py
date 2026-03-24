"""Sketched-Isotropic-Gaussian Regularizer (SIGReg).

Enforces latent embeddings to match an isotropic Gaussian N(0, I) by:
1. Projecting embeddings onto M random unit-norm directions (Cramér-Wold)
2. Computing the Epps-Pulley normality test statistic on each 1-D projection
3. Averaging over all projections

Reference:
    Balestriero & LeCun, "LeJEPA: Provable and Scalable Self-Supervised
    Learning without the Heuristics", 2025.  (arXiv:2511.08544)
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def _epps_pulley(
    h: torch.Tensor,
    t_nodes: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    """Epps-Pulley test statistic for a batch of 1-D projections.

    Measures the squared integrated difference between the empirical
    characteristic function (ECF) of *h* and the CF of N(0, 1).

    Args:
        h: (B, M) — B samples for each of M projections.
        t_nodes: (K,) — quadrature nodes.
        weights: (K,) — trapezoid quadrature weights.

    Returns:
        (M,) — test statistic per projection.
    """
    # th: (B, M, K)
    th = h.unsqueeze(-1) * t_nodes.view(1, 1, -1)

    # Empirical CF: φ_N(t) = (1/B) Σ_n exp(i·t·h_n)
    ecf_real = th.cos().mean(dim=0)          # (M, K)
    ecf_imag = th.sin().mean(dim=0)          # (M, K)

    # Target CF for N(0,1): φ_0(t) = exp(-t²/2)
    phi0 = torch.exp(-0.5 * t_nodes * t_nodes)  # (K,)

    # |φ_N(t) - φ_0(t)|²  (φ_0 is real-valued)
    diff_sq = (ecf_real - phi0.unsqueeze(0)).square() + ecf_imag.square()

    # Weighted trapezoid integral  →  (M,)
    return (diff_sq * weights.unsqueeze(0)).sum(dim=-1)


def sigreg(
    Z: torch.Tensor,
    n_projections: int = 1024,
    n_knots: int = 17,
    t_min: float = 0.2,
    t_max: float = 4.0,
) -> torch.Tensor:
    """Compute the SIGReg loss for a batch of embeddings.

    Encourages the distribution of *Z* to match N(0, I_D) by testing
    normality along random 1-D slices (Cramér-Wold theorem).

    Args:
        Z: (B, D) batch of latent embeddings.
        n_projections: number of random unit-norm directions M.
        n_knots: number of trapezoid quadrature nodes K.
        t_min: left endpoint of integration interval.
        t_max: right endpoint of integration interval.

    Returns:
        Scalar SIGReg loss (lower ≈ closer to isotropic Gaussian).
    """
    B, D = Z.shape

    # Random unit-norm directions on S^{D-1}
    u = torch.randn(D, n_projections, device=Z.device, dtype=Z.dtype)
    u = F.normalize(u, dim=0)

    # Project embeddings:  h^(m) = Z @ u^(m),  shape (B, M)
    h = Z @ u

    # Quadrature nodes and trapezoid weights
    t_nodes = torch.linspace(t_min, t_max, n_knots, device=Z.device, dtype=Z.dtype)
    dt = (t_max - t_min) / (n_knots - 1)
    weights = torch.full((n_knots,), dt, device=Z.device, dtype=Z.dtype)
    weights[0]  *= 0.5
    weights[-1] *= 0.5

    # Average Epps-Pulley statistic over all projections
    return _epps_pulley(h, t_nodes, weights).mean()


def sigreg_stepwise(
    Z_seq: torch.Tensor,
    n_projections: int = 1024,
    n_knots: int = 17,
    t_min: float = 0.2,
    t_max: float = 4.0,
) -> torch.Tensor:
    """Step-wise SIGReg: average SIGReg independently at each time step.

    Args:
        Z_seq: (B, T, D) — embeddings over a temporal sequence.

    Returns:
        Scalar loss (mean over T time-steps).
    """
    B, T, D = Z_seq.shape
    total = torch.tensor(0.0, device=Z_seq.device, dtype=Z_seq.dtype)
    for t in range(T):
        total = total + sigreg(Z_seq[:, t], n_projections, n_knots, t_min, t_max)
    return total / T
