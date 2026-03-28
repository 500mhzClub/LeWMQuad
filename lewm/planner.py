"""CEM planner over latent energy landscape.

Scores candidate action sequences by rolling them out through the
world model predictor and summing per-step energy from the
LatentEnergyHead.  Supports warm-starting (MPC) and optional
goal-matching cost blending.

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

import torch
import torch.nn as nn


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
    stall_penalty: float = 0.5
    # Momentum for warm-start blending (0 = pure warm-start, 1 = pure re-init)
    warmstart_decay: float = 0.0
    # Optional goal-matching weight (0 = energy-only)
    goal_weight: float = 0.0


class CEMPlanner:
    """Cross-Entropy Method planner over the latent energy landscape.

    Each planning call:
      1. Sample N action sequences from N(mean, std).
      2. Clamp to action bounds.
      3. Rollout through the world model predictor.
      4. Score via energy head (+ optional goal cost).
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
            z_raw: (1, D) or (D,) — raw encoder embedding of current frame.
            z_goal_proj: optional (1, D) or (D,) — projected goal embedding.

        Returns:
            best_actions: (H, action_dim) — planned action sequence.
            best_cost:    scalar — cost of the best trajectory.
        """
        if z_raw.dim() == 1:
            z_raw = z_raw.unsqueeze(0)
        if z_goal_proj is not None and z_goal_proj.dim() == 1:
            z_goal_proj = z_goal_proj.unsqueeze(0)

        H = self.cfg.horizon
        N = self.cfg.n_candidates
        K = self.cfg.n_elites
        D = self.cfg.action_dim

        # Initialise distribution
        if self._prev_mean is not None:
            # Shift forward by one step, repeat last action
            mean = torch.cat([self._prev_mean[1:], self._prev_mean[-1:]], dim=0)
            # Blend toward zero based on decay
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

            # Rollout
            z_pred = self.wm.plan_rollout(z_start, actions)  # (N, H, D_latent)

            # Energy cost
            costs = self.eh.score_trajectory(z_pred)  # (N,)

            # Forward-velocity bonus (encourages exploration)
            if self.cfg.stall_penalty > 0:
                fwd_speed = actions[:, :, 0].mean(dim=1)  # avg vx over horizon
                costs = costs - self.cfg.stall_penalty * fwd_speed

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
            vis: (1, C, H, W) or (C, H, W) — current frame (float, [0,1]).
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
            (action_dim,) — action to execute this timestep.
        """
        action_seq, _ = self.plan_from_obs(vis, proprio, goal_vis, goal_proprio)
        return action_seq[0]

    def reset(self):
        """Clear warm-start state (call between episodes)."""
        self._prev_mean = None
