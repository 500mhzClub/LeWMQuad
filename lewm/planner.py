"""CEM planner over latent energy landscape with exploration.

Scores candidate action sequences by rolling them out through the
world model predictor and scoring them with the LatentEnergyHead.
A latent coverage grid rewards trajectories that visit novel regions
of representation space, but only when those trajectories also
translate through space rather than spinning in place. A stuck detector
forces recovery turns when the robot's observations or recent pose
history stop changing enough.

Typical usage::

    planner = CEMPlanner(world_model, energy_head, device=device)

    # MPC loop
    for obs in environment:
        action = planner.step(obs.vision, obs.proprio)
        env.apply(action)
    planner.reset()
"""
from __future__ import annotations

import random as pyrandom
from collections import deque
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class CEMConfig:
    """All CEM hyper-parameters in one place."""
    horizon: int = 15
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
    forward_bias: Tuple[float, float, float] = (0.25, 0.0, 0.0)
    # Cost component weights
    energy_weight: float = 1.0     # mean per-step energy
    stall_penalty: float = 1.0     # forward progress reward in free space
    coverage_weight: float = 0.3   # mild novelty drive
    yaw_penalty: float = 0.35      # discourage spin-in-place when free
    coverage_motion_scale: float = 0.20  # speed needed for full novelty credit
    # Collision-reactive planning (when sim reports wall contact)
    collision_fwd_penalty: float = 8.0   # penalize vx>0 when colliding
    collision_yaw_bonus: float = 4.0     # reward |yaw_rate| when colliding
    # Latent coverage grid (random-projection hash)
    coverage_dim: int = 3
    coverage_cells: int = 8        # 8^3 = 512 cells
    # Stuck detection
    stuck_window: int = 25
    stuck_threshold: float = 0.02
    stuck_patience: int = 8
    # Pose-based local stagnation detection
    position_window: int = 60
    position_radius: float = 0.35
    # Recovery turn parameters
    turn_steps_min: int = 15
    turn_steps_max: int = 45
    turn_yaw_min: float = 0.8
    turn_yaw_max: float = 1.4
    # Fraction of recovery spent reversing before turning
    reverse_fraction: float = 0.4
    reverse_speed: float = -0.3
    escape_forward_speed: float = 0.25
    # Grace period after recovery before stuck detection resumes
    post_recovery_grace: int = 15
    # Momentum for warm-start
    warmstart_decay: float = 0.0
    # Optional goal-matching weight
    goal_weight: float = 0.0


# ---------------------------------------------------------------------- #
# Latent coverage grid
# ---------------------------------------------------------------------- #

class LatentCoverageGrid:
    """Hash-grid coverage map over latent space."""

    def __init__(
        self,
        latent_dim: int,
        proj_dim: int = 3,
        n_cells: int = 8,
        device: torch.device = torch.device("cpu"),
        seed: int = 42,
    ):
        self.device = device
        self.proj_dim = proj_dim
        self.n_cells = n_cells
        self.latent_dim = latent_dim

        rng = torch.Generator(device="cpu").manual_seed(seed)
        W = torch.randn(latent_dim, proj_dim, generator=rng, device="cpu")
        Q, _ = torch.linalg.qr(W)
        self.proj = Q.to(device)

        self._proj_min = torch.full((proj_dim,), float("inf"), device=device)
        self._proj_max = torch.full((proj_dim,), float("-inf"), device=device)
        self._counts: dict[int, int] = {}
        self._total_visits = 0

    def _project(self, z: torch.Tensor) -> torch.Tensor:
        return z.float() @ self.proj

    def _to_cell_ids(self, p: torch.Tensor) -> torch.Tensor:
        rng = (self._proj_max - self._proj_min).clamp(min=1e-6)
        normed = (p - self._proj_min) / rng
        bins = normed.clamp(0.0, 0.999).mul(self.n_cells).long()
        multipliers = self.n_cells ** torch.arange(
            self.proj_dim, device=self.device, dtype=torch.long
        )
        return (bins * multipliers).sum(dim=-1)

    def mark(self, z: torch.Tensor):
        if z.dim() == 1:
            z = z.unsqueeze(0)
        p = self._project(z)
        self._proj_min = torch.min(self._proj_min, p.squeeze(0))
        self._proj_max = torch.max(self._proj_max, p.squeeze(0))
        cell_id = int(self._to_cell_ids(p).item())
        self._counts[cell_id] = self._counts.get(cell_id, 0) + 1
        self._total_visits += 1

    def novelty_batch(self, z_seq: torch.Tensor) -> torch.Tensor:
        if self._total_visits == 0:
            return torch.zeros(z_seq.shape[0], device=self.device)
        N, H, D = z_seq.shape
        p = self._project(z_seq.reshape(N * H, D))
        cell_ids = self._to_cell_ids(p)
        counts = torch.tensor(
            [self._counts.get(int(c), 0) for c in cell_ids.cpu().tolist()],
            device=self.device, dtype=torch.float32,
        )
        return (1.0 / (1.0 + counts)).reshape(N, H).sum(dim=1)

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


# ---------------------------------------------------------------------- #
# Stuck detector
# ---------------------------------------------------------------------- #

class StuckDetector:
    """Detects when the robot is stuck by monitoring latent observation velocity."""

    def __init__(self, window: int = 25, threshold: float = 0.02, patience: int = 8):
        self.window = window
        self.threshold = threshold
        self.patience = patience
        self._latents: deque = deque(maxlen=window)
        self._stuck_count: int = 0
        self.total_stuck_events: int = 0

    def update(self, z: torch.Tensor) -> bool:
        self._latents.append(z.detach().float().cpu().squeeze())
        if len(self._latents) < self.window:
            return False
        z_stack = torch.stack(list(self._latents))
        avg_delta = (z_stack[1:] - z_stack[:-1]).norm(dim=-1).mean().item()
        if avg_delta < self.threshold:
            self._stuck_count += 1
        else:
            self._stuck_count = 0
        triggered = self._stuck_count >= self.patience
        if triggered:
            self.total_stuck_events += 1
            self._stuck_count = 0
        return triggered

    def reset(self):
        self._latents.clear()
        self._stuck_count = 0
        self.total_stuck_events = 0


# ---------------------------------------------------------------------- #
# CEM Planner
# ---------------------------------------------------------------------- #

class CEMPlanner:
    """CEM planner with JEPA rollouts, latent coverage, and collision-aware recovery.

    Normal operation:
      1. Sample N action sequences from N(mean, std).
      2. Rollout through the world model predictor.
      3. Score: energy_weight * mean_energy - stall_penalty * fwd_speed
                - coverage_weight * motion-gated novelty
                + yaw penalty in free space
                + collision-reactive terms when contact is reported
      4. Refit from elites. Repeat.

    When the stuck detector fires (latent observations stopped changing):
      Execute an alternating recovery turn for 15-45 steps, bypassing CEM.
      This breaks out of dead ends where all CEM candidates look the same.
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

        self._lo = torch.tensor(self.cfg.action_low, device=self.device)
        self._hi = torch.tensor(self.cfg.action_high, device=self.device)
        self._fwd_bias = torch.tensor(self.cfg.forward_bias, device=self.device)

        # Warm-start
        self._prev_mean: Optional[torch.Tensor] = None

        # Coverage grid (lazy init)
        self._coverage: Optional[LatentCoverageGrid] = None

        # Stuck detection + recovery turn state
        self._stuck = StuckDetector(
            window=self.cfg.stuck_window,
            threshold=self.cfg.stuck_threshold,
            patience=self.cfg.stuck_patience,
        )
        self._is_stuck: bool = False
        self._turn_remaining: int = 0
        self._turn_total: int = 0       # total steps in current recovery
        self._turn_reverse_steps: int = 0
        self._turn_yaw: float = 0.0
        self._grace_remaining: int = 0  # post-recovery grace period

        # Collision-based recovery (proprioceptive — robot feels bumps)
        self._collision_window: deque = deque(maxlen=10)
        self._collision_threshold: int = 4  # 4+ collisions in 10 steps = stuck in wall

        # Alternating recovery direction
        self._last_turn_sign: float = 1.0

        # Pose history for local stagnation detection
        self._position_window: deque = deque(maxlen=self.cfg.position_window)

        # Diagnostics from last CEM iteration
        self.diag: dict = {}

    def _ensure_coverage(self, latent_dim: int):
        if self._coverage is None:
            self._coverage = LatentCoverageGrid(
                latent_dim=latent_dim,
                proj_dim=self.cfg.coverage_dim,
                n_cells=self.cfg.coverage_cells,
                device=self.device,
            )

    # ------------------------------------------------------------------ #
    # Core CEM planning
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def plan(
        self,
        z_raw: torch.Tensor,
        z_goal_proj: Optional[torch.Tensor] = None,
        colliding: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """CEM planning from an encoded latent using JEPA rollouts."""
        if z_raw.dim() == 1:
            z_raw = z_raw.unsqueeze(0)
        if z_goal_proj is not None and z_goal_proj.dim() == 1:
            z_goal_proj = z_goal_proj.unsqueeze(0)

        self._ensure_coverage(z_raw.shape[-1])
        self._coverage.mark(z_raw)

        H = self.cfg.horizon
        N = self.cfg.n_candidates
        K = self.cfg.n_elites
        D = self.cfg.action_dim

        # Initialise distribution
        # Clear warm-start when colliding — don't keep planning into walls
        if colliding:
            self._prev_mean = None

        if self._prev_mean is not None:
            mean = torch.cat([self._prev_mean[1:], self._prev_mean[-1:]], dim=0)
            if self.cfg.warmstart_decay > 0:
                mean = mean * (1.0 - self.cfg.warmstart_decay)
        else:
            if colliding:
                mean = torch.zeros(H, D, device=self.device)
            else:
                mean = self._fwd_bias.unsqueeze(0).expand(H, -1).clone()
        std = torch.full((H, D), self.cfg.init_std, device=self.device)

        z_start = z_raw.expand(N, -1)

        for _it in range(self.cfg.n_iterations):
            noise = torch.randn(N, H, D, device=self.device)
            actions = (mean.unsqueeze(0) + std.unsqueeze(0) * noise).clamp(
                self._lo, self._hi,
            )

            # JEPA rollout
            z_pred = self.wm.plan_rollout(z_start, actions)

            # Mean per-step energy keeps horizon scaling comparable to the
            # command-based terms below.
            energy = self.eh.score_trajectory(z_pred) / float(H)
            costs = self.cfg.energy_weight * energy

            # Forward-velocity bonus
            if self.cfg.stall_penalty > 0:
                costs = costs - self.cfg.stall_penalty * actions[:, :, 0].mean(dim=1)

            # Latent coverage bonus
            if self.cfg.coverage_weight > 0 and self._coverage.total_visits > 0:
                novelty = self._coverage.novelty_batch(z_pred) / float(H)
                lin_speed = actions[:, :, :2].norm(dim=-1).mean(dim=1)
                motion_gate = (lin_speed / max(self.cfg.coverage_motion_scale, 1e-6)).clamp(0.0, 1.0)
                costs = costs - self.cfg.coverage_weight * novelty * motion_gate

            # Free-space yaw penalty: latent novelty can otherwise be hacked by
            # rotating in place and changing the camera view without exploring.
            if not colliding and self.cfg.yaw_penalty > 0:
                costs = costs + self.cfg.yaw_penalty * actions[:, :, 2].abs().mean(dim=1)

            # Collision-reactive: penalize forward, reward turning
            if colliding:
                fwd = actions[:, :, 0].clamp(min=0).mean(dim=1)
                costs = costs + self.cfg.collision_fwd_penalty * fwd
                yaw = actions[:, :, 2].abs().mean(dim=1)
                costs = costs - self.cfg.collision_yaw_bonus * yaw

            # Goal cost
            if z_goal_proj is not None and self.cfg.goal_weight > 0:
                goal_cost = self.wm.plan_cost(z_pred, z_goal_proj.expand(N, -1))
                costs = costs + self.cfg.goal_weight * goal_cost

            elite_idx = costs.topk(K, largest=False).indices
            elite_actions = actions[elite_idx]
            mean = elite_actions.mean(dim=0)
            std = elite_actions.std(dim=0).clamp(min=self.cfg.min_std)

        self._prev_mean = mean.clone()

        # Capture diagnostics for debugging
        self.diag = {
            "cost_mean": float(costs.mean()),
            "cost_std": float(costs.std()),
            "cost_min": float(costs.min()),
            "cost_max": float(costs.max()),
            "energy_mean": float(energy.mean()),
            "energy_std": float(energy.std()),
            "z_pred_std": float(z_pred.std()),
            "elite_vx": float(elite_actions[:, :, 0].mean()),
            "elite_yaw": float(elite_actions[:, :, 2].mean()),
        }

        return mean, costs[elite_idx[0]]

    # ------------------------------------------------------------------ #
    # Step: CEM + stuck recovery
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def step(
        self,
        vis: torch.Tensor,
        proprio: torch.Tensor | None,
        goal_vis: Optional[torch.Tensor] = None,
        goal_proprio: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Plan one action: CEM when free, recovery turn when stuck."""
        if vis.dim() == 3:
            vis = vis.unsqueeze(0)
        if proprio is not None and proprio.dim() == 1:
            proprio = proprio.unsqueeze(0)

        z_raw = self.wm.encode_raw(vis, proprio)

        # If in recovery turn, bypass CEM and skip stuck detection entirely
        if self._turn_remaining > 0:
            self._is_stuck = False  # only True on the detection step
            self._turn_remaining -= 1
            self._ensure_coverage(z_raw.shape[-1])
            self._coverage.mark(z_raw)

            # First phase: reverse to back away from obstacle
            steps_done = self._turn_total - self._turn_remaining - 1
            if steps_done < self._turn_reverse_steps:
                cmd = torch.tensor(
                    [self.cfg.reverse_speed, 0.0, 0.0], device=self.device,
                )
            else:
                # Second phase: forward + turn
                cmd = torch.tensor(
                    [self.cfg.escape_forward_speed, 0.0, self._turn_yaw], device=self.device,
                )

            if self._turn_remaining == 0:
                # Recovery just ended — start grace period
                self._grace_remaining = self.cfg.post_recovery_grace
                self._collision_window.clear()
                self._position_window.clear()
                self._stuck._latents.clear()
                self._stuck._stuck_count = 0
            return cmd.clamp(self._lo, self._hi)

        # Grace period: skip stuck detection but run CEM normally
        if self._grace_remaining > 0:
            self._grace_remaining -= 1
            self._stuck.update(z_raw)  # feed data but ignore result
            self._is_stuck = False
        else:
            # Stuck detection: latent velocity OR collision rate
            latent_stuck = self._stuck.update(z_raw)
            collision_stuck = self._collision_stuck()
            position_stuck = self._position_stuck()
            self._is_stuck = latent_stuck or collision_stuck or position_stuck
            if self._is_stuck:
                self._turn_total = pyrandom.randint(
                    self.cfg.turn_steps_min, self.cfg.turn_steps_max,
                )
                self._turn_remaining = self._turn_total
                self._turn_reverse_steps = (
                    int(self._turn_total * self.cfg.reverse_fraction)
                    if collision_stuck else 0
                )
                self._turn_yaw = self._last_turn_sign * pyrandom.uniform(
                    self.cfg.turn_yaw_min, self.cfg.turn_yaw_max,
                )
                self._last_turn_sign *= -1.0  # alternate for next recovery
                self._prev_mean = None  # clear warm-start
                if not latent_stuck and (collision_stuck or position_stuck):
                    self._stuck.total_stuck_events += 1
                # Immediately start recovery on this step (fall through to
                # the recovery branch on the next call; for this step, just
                # issue the first reverse command directly)
                self._is_stuck = True  # signal to eval logger for this one step
                self._turn_remaining -= 1
                self._ensure_coverage(z_raw.shape[-1])
                self._coverage.mark(z_raw)
                cmd = torch.tensor(
                    [
                        self.cfg.reverse_speed if self._turn_reverse_steps > 0 else self.cfg.escape_forward_speed,
                        0.0,
                        0.0 if self._turn_reverse_steps > 0 else self._turn_yaw,
                    ],
                    device=self.device,
                )
                return cmd.clamp(self._lo, self._hi)

        # Normal CEM planning
        z_goal_proj = None
        if goal_vis is not None:
            if goal_vis.dim() == 3:
                goal_vis = goal_vis.unsqueeze(0)
            if goal_proprio is not None and goal_proprio.dim() == 1:
                goal_proprio = goal_proprio.unsqueeze(0)
            z_goal_proj = self.wm.encode_observation(goal_vis, goal_proprio)

        recently_colliding = self._is_colliding_recently()
        action_seq, _ = self.plan(z_raw, z_goal_proj, colliding=recently_colliding)
        return action_seq[0]

    # ------------------------------------------------------------------ #
    # Properties
    # ------------------------------------------------------------------ #

    def report_collision(self, colliding: bool):
        """Feed collision signal from the sim (proprioceptive feedback)."""
        self._collision_window.append(colliding)

    def report_pose(self, xy: np.ndarray | torch.Tensor):
        """Feed world-frame XY pose for local stagnation detection."""
        if isinstance(xy, torch.Tensor):
            xy = xy.detach().float().cpu().numpy()
        xy = np.asarray(xy, dtype=np.float32).reshape(-1)
        if xy.shape[0] != 2:
            raise ValueError(f"Expected XY pose with shape (2,), got {xy.shape}")
        self._position_window.append(xy.copy())

    def _collision_stuck(self) -> bool:
        """True if the robot is colliding too frequently (stuck in wall)."""
        if len(self._collision_window) < self._collision_window.maxlen:
            return False
        return sum(self._collision_window) >= self._collision_threshold

    def _is_colliding_recently(self) -> bool:
        """True if colliding frequently in recent steps (for reactive CEM cost)."""
        recent = list(self._collision_window)[-3:]
        return len(recent) >= 3 and sum(recent) >= 2

    def _position_stuck(self) -> bool:
        """True if recent poses stay inside a small radius."""
        if len(self._position_window) < self._position_window.maxlen:
            return False
        pts = np.stack(self._position_window, axis=0)
        center = pts.mean(axis=0, keepdims=True)
        radius = np.linalg.norm(pts - center, axis=1).max()
        return bool(radius < self.cfg.position_radius)

    @property
    def coverage(self) -> Optional[LatentCoverageGrid]:
        return self._coverage

    @property
    def is_stuck(self) -> bool:
        return self._is_stuck

    @property
    def is_turning(self) -> bool:
        return self._turn_remaining > 0

    @property
    def stuck_events(self) -> int:
        return self._stuck.total_stuck_events

    def reset(self):
        """Clear all state between episodes."""
        self._prev_mean = None
        self._is_stuck = False
        self._turn_remaining = 0
        self._turn_total = 0
        self._turn_reverse_steps = 0
        self._turn_yaw = 0.0
        self._last_turn_sign = 1.0
        self._grace_remaining = 0
        self._collision_window.clear()
        self._position_window.clear()
        self.diag = {}
        if self._coverage is not None:
            self._coverage.clear()
        self._stuck.reset()
