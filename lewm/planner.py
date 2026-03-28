"""CEM planner over latent energy landscape with latent coverage.

Scores candidate action sequences by rolling them out through the
world model predictor and summing per-step energy from the
LatentEnergyHead.  A latent coverage map (random-projection hash grid)
rewards trajectories that visit novel regions of representation space,
driving exploration without access to ground-truth position.

Typical usage::

    planner = CEMPlanner(world_model, energy_head, device=device)

    # MPC loop
    for obs in environment:
        action = planner.step(obs.vision, obs.proprio)
        env.apply(action)
    planner.reset()
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class CEMConfig:
    """All CEM hyper-parameters in one place."""
    horizon: int = 8
    n_candidates: int = 128
    n_elites: int = 16
    n_iterations: int = 3
    action_dim: int = 3
    # Per-dimension action bounds [vx, vy, yaw_rate]
    action_low: Tuple[float, float, float] = (-0.5, -0.3, -1.5)
    action_high: Tuple[float, float, float] = (0.5, 0.3, 1.5)
    init_std: float = 0.5
    min_std: float = 0.01
    # Forward-motion bias for initial CEM mean (vx, vy, yaw_rate)
    forward_bias: Tuple[float, float, float] = (0.3, 0.0, 0.0)
    # Penalty weight for low forward velocity (encourages exploration)
    stall_penalty: float = 2.0
    # Latent coverage grid (random-projection hash)
    coverage_weight: float = 1.0
    coverage_dim: int = 8          # dimensionality of the hash projection
    coverage_cells: int = 16       # bins per projected dimension
    # Momentum for warm-start blending (0 = pure warm-start, 1 = pure re-init)
    warmstart_decay: float = 0.0
    # Optional goal-matching weight (0 = energy-only)
    goal_weight: float = 0.0


class LatentCoverageGrid:
    """Hash-grid coverage map over latent space.

    Projects high-dimensional latents through a fixed random matrix into
    a low-dimensional space, then discretises into grid cells.  Each cell
    tracks a visit count.  Trajectories passing through rarely-visited
    cells get a novelty bonus.

    This gives position-like discrimination without ground-truth coordinates:
    the random projection amplifies small latent differences that cosine
    similarity would miss, and the discretisation creates hard boundaries
    that prevent "everything looks the same" collapse.
    """

    def __init__(
        self,
        latent_dim: int,
        proj_dim: int = 8,
        n_cells: int = 16,
        device: torch.device = torch.device("cpu"),
        seed: int = 42,
    ):
        self.device = device
        self.proj_dim = proj_dim
        self.n_cells = n_cells
        self.latent_dim = latent_dim

        # Fixed random projection (orthogonal-ish via QR, built on CPU for reproducibility)
        rng = torch.Generator(device="cpu").manual_seed(seed)
        W = torch.randn(latent_dim, proj_dim, generator=rng, device="cpu")
        Q, _ = torch.linalg.qr(W)
        self.proj = Q.to(device)  # (latent_dim, proj_dim)

        # We'll track min/max of projected values to auto-scale bins
        self._proj_min = torch.full((proj_dim,), float("inf"), device=device)
        self._proj_max = torch.full((proj_dim,), float("-inf"), device=device)

        # Visit count table: flat hash -> count
        # Use a dict for sparse storage (grid could be 16^8 = 4B cells)
        self._counts: dict[int, int] = {}
        self._total_visits = 0

    def _project(self, z: torch.Tensor) -> torch.Tensor:
        """Project latents to low-dim space. z: (..., D) -> (..., proj_dim)."""
        return z.float() @ self.proj

    def _to_cell_ids(self, p: torch.Tensor) -> torch.Tensor:
        """Convert projected vectors to flat cell IDs. p: (..., proj_dim) -> (...,)."""
        # Normalise each dimension to [0, 1] using running min/max
        rng = (self._proj_max - self._proj_min).clamp(min=1e-6)
        normed = (p - self._proj_min) / rng  # (..., proj_dim)
        # Discretise to [0, n_cells-1]
        bins = normed.clamp(0.0, 0.999).mul(self.n_cells).long()  # (..., proj_dim)
        # Flat hash via mixed-radix encoding
        multipliers = self.n_cells ** torch.arange(
            self.proj_dim, device=self.device, dtype=torch.long
        )
        return (bins * multipliers).sum(dim=-1)  # (...,)

    def mark(self, z: torch.Tensor):
        """Record a visit for a single observation. z: (1, D) or (D,)."""
        if z.dim() == 1:
            z = z.unsqueeze(0)
        p = self._project(z)  # (1, proj_dim)
        # Update running stats
        self._proj_min = torch.min(self._proj_min, p.squeeze(0))
        self._proj_max = torch.max(self._proj_max, p.squeeze(0))

        cell_id = int(self._to_cell_ids(p).item())
        self._counts[cell_id] = self._counts.get(cell_id, 0) + 1
        self._total_visits += 1

    def novelty_batch(self, z_seq: torch.Tensor) -> torch.Tensor:
        """Score trajectory novelty by visit counts of predicted latent cells.

        Args:
            z_seq: (N, H, D) -- predicted latent trajectories.

        Returns:
            (N,) -- summed novelty bonus.  Unvisited cells contribute 1.0,
            frequently visited cells contribute 1/(1+count).
        """
        if self._total_visits == 0:
            return torch.zeros(z_seq.shape[0], device=self.device)

        N, H, D = z_seq.shape
        p = self._project(z_seq.reshape(N * H, D))  # (N*H, proj_dim)
        cell_ids = self._to_cell_ids(p)               # (N*H,)

        # Look up counts
        counts = torch.tensor(
            [self._counts.get(int(c), 0) for c in cell_ids.cpu().tolist()],
            device=self.device, dtype=torch.float32,
        )
        novelty = (1.0 / (1.0 + counts)).reshape(N, H).sum(dim=1)
        return novelty

    @property
    def cells_visited(self) -> int:
        return len(self._counts)

    @property
    def total_visits(self) -> int:
        return self._total_visits

    def clear(self):
        self._counts.clear()
        self._total_visits = 0
        self._proj_min.fill_(float("inf"))
        self._proj_max.fill_(float("-inf"))


class CEMPlanner:
    """Cross-Entropy Method planner over the latent energy landscape.

    Each planning call:
      1. Sample N action sequences from N(mean, std).
      2. Clamp to action bounds.
      3. Rollout through the world model predictor.
      4. Score via energy head + coverage bonus + stall penalty.
      5. Refit distribution from top-K elites.
      6. Repeat for ``n_iterations``.

    Warm-starting: the previous solution is time-shifted by one step and
    used as the initial mean for the next call, reducing re-planning cost.
    """

    def __init__(
        self,
        world_model: nn.Module,
        energy_head: nn.Module,
        config: CEMConfig | None = None,
        device: torch.device | str = "cuda",
    ):
        self.wm = world_model
        self.eh = energy_head
        self.cfg = config or CEMConfig()
        self.device = torch.device(device)

        # Pre-compute bound tensors
        self._lo = torch.tensor(self.cfg.action_low, device=self.device)
        self._hi = torch.tensor(self.cfg.action_high, device=self.device)
        self._fwd_bias = torch.tensor(self.cfg.forward_bias, device=self.device)

        # Warm-start state
        self._prev_mean: Optional[torch.Tensor] = None

        # Latent coverage grid (initialised lazily once latent_dim is known)
        self._coverage: Optional[LatentCoverageGrid] = None

    def _ensure_coverage(self, latent_dim: int):
        """Lazily create the coverage grid once we know the latent dim."""
        if self._coverage is None:
            self._coverage = LatentCoverageGrid(
                latent_dim=latent_dim,
                proj_dim=self.cfg.coverage_dim,
                n_cells=self.cfg.coverage_cells,
                device=self.device,
            )

    # ------------------------------------------------------------------ #
    # Core planning
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def plan(
        self,
        z_raw: torch.Tensor,
        z_goal_proj: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Plan from an already-encoded latent.

        Args:
            z_raw: (1, D) or (D,) -- raw encoder embedding of current frame.
            z_goal_proj: optional (1, D) or (D,) -- projected goal embedding.

        Returns:
            best_actions: (H, action_dim) -- planned action sequence.
            best_cost:    scalar -- cost of the best trajectory.
        """
        if z_raw.dim() == 1:
            z_raw = z_raw.unsqueeze(0)
        if z_goal_proj is not None and z_goal_proj.dim() == 1:
            z_goal_proj = z_goal_proj.unsqueeze(0)

        # Mark current latent in coverage grid
        self._ensure_coverage(z_raw.shape[-1])
        self._coverage.mark(z_raw)

        H = self.cfg.horizon
        N = self.cfg.n_candidates
        K = self.cfg.n_elites
        D = self.cfg.action_dim

        # Initialise distribution
        if self._prev_mean is not None:
            # Shift forward by one step, repeat last action
            mean = torch.cat([self._prev_mean[1:], self._prev_mean[-1:]], dim=0)
            if self.cfg.warmstart_decay > 0:
                mean = mean * (1.0 - self.cfg.warmstart_decay)
        else:
            mean = self._fwd_bias.unsqueeze(0).expand(H, -1).clone()
        std = torch.full((H, D), self.cfg.init_std, device=self.device)

        # Expand z_start for all candidates: (1, D) -> (N, D)
        z_start = z_raw.expand(N, -1)

        for _it in range(self.cfg.n_iterations):
            # Sample
            noise = torch.randn(N, H, D, device=self.device)
            actions = mean.unsqueeze(0) + std.unsqueeze(0) * noise
            actions = actions.clamp(self._lo, self._hi)

            # Rollout through world model
            z_pred = self.wm.plan_rollout(z_start, actions)  # (N, H, D_latent)

            # Energy cost (lower = safer)
            costs = self.eh.score_trajectory(z_pred)  # (N,)

            # Forward-velocity bonus
            if self.cfg.stall_penalty > 0:
                fwd_speed = actions[:, :, 0].mean(dim=1)
                costs = costs - self.cfg.stall_penalty * fwd_speed

            # Latent coverage bonus
            if self.cfg.coverage_weight > 0 and self._coverage.total_visits > 0:
                cov_bonus = self._coverage.novelty_batch(z_pred)  # (N,)
                costs = costs - self.cfg.coverage_weight * cov_bonus

            # Optional goal cost
            if z_goal_proj is not None and self.cfg.goal_weight > 0:
                goal_cost = self.wm.plan_cost(z_pred, z_goal_proj.expand(N, -1))
                costs = costs + self.cfg.goal_weight * goal_cost

            # Select elites
            elite_idx = costs.topk(K, largest=False).indices
            elite_actions = actions[elite_idx]  # (K, H, D)

            # Refit
            mean = elite_actions.mean(dim=0)
            std = elite_actions.std(dim=0).clamp(min=self.cfg.min_std)

        # Cache for warm-start
        self._prev_mean = mean.clone()

        best_cost = costs[elite_idx[0]]
        return mean, best_cost

    # ------------------------------------------------------------------ #
    # Convenience: encode + plan in one call
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def plan_from_obs(
        self,
        vis: torch.Tensor,
        proprio: torch.Tensor | None,
        goal_vis: Optional[torch.Tensor] = None,
        goal_proprio: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode current (and optional goal) observation, then plan.

        Args:
            vis: (1, C, H, W) or (C, H, W) -- current frame (float, [0,1]).
            proprio: (1, P) or (P,) or None.
            goal_vis: optional goal frame.
            goal_proprio: optional goal proprioception.

        Returns:
            best_actions: (H, action_dim)
            best_cost: scalar
        """
        if vis.dim() == 3:
            vis = vis.unsqueeze(0)
        if proprio is not None and proprio.dim() == 1:
            proprio = proprio.unsqueeze(0)

        z_raw = self.wm.encode_raw(vis, proprio)

        z_goal_proj = None
        if goal_vis is not None:
            if goal_vis.dim() == 3:
                goal_vis = goal_vis.unsqueeze(0)
            if goal_proprio is not None and goal_proprio.dim() == 1:
                goal_proprio = goal_proprio.unsqueeze(0)
            z_goal_proj = self.wm.encode_observation(goal_vis, goal_proprio)

        return self.plan(z_raw, z_goal_proj)

    def step(
        self,
        vis: torch.Tensor,
        proprio: torch.Tensor | None,
        goal_vis: Optional[torch.Tensor] = None,
        goal_proprio: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Plan and return only the first action (for MPC loop).

        Returns:
            (action_dim,) -- action to execute this timestep.
        """
        action_seq, _ = self.plan_from_obs(vis, proprio, goal_vis, goal_proprio)
        return action_seq[0]

    @property
    def coverage(self) -> Optional[LatentCoverageGrid]:
        """Access the coverage grid (for HUD rendering / metrics)."""
        return self._coverage

    def reset(self):
        """Clear warm-start state and coverage grid (call between episodes)."""
        self._prev_mean = None
        if self._coverage is not None:
            self._coverage.clear()
