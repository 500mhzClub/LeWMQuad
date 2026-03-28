"""Greedy one-step energy planner with contact escape.

Scores a small set of discrete action primitives using one-step
predictions through the world model, avoiding the long-horizon
autoregressive rollouts that suffer from exposure bias and
wall-clipping hallucinations.

Two-mode controller:
  - **Free mode** (no collision): encode current observation, predict
    one step for each action primitive, score by energy + beacon
    proximity + frontier bonus, take the argmin.
  - **Contact mode** (collision detected): bypass JEPA entirely
    (camera is clipped through the wall), execute a deterministic
    reverse-then-turn escape sequence.

Typical usage::

    planner = GreedyEnergyPlanner(world_model, energy_head, device=device)
    planner.set_beacon_targets(beacon_latents)

    for obs in environment:
        action = planner.step(obs.vision, obs.proprio, obs.colliding)
        env.apply(action)
    planner.reset()
"""
from __future__ import annotations

import math
import random as pyrandom
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import torch
import torch.nn as nn


# ---------------------------------------------------------------------- #
# Configuration
# ---------------------------------------------------------------------- #

# Discrete action primitives: (vx, vy, yaw_rate)
DEFAULT_ACTION_SET: List[Tuple[float, float, float]] = [
    ( 0.35,  0.00,  0.0),    # forward
    ( 0.25,  0.00,  0.4),    # forward-left
    ( 0.25,  0.00, -0.4),    # forward-right
    ( 0.15,  0.00,  0.8),    # sharp left
    ( 0.15,  0.00, -0.8),    # sharp right
    ( 0.00,  0.00,  1.2),    # spin left
    ( 0.00,  0.00, -1.2),    # spin right
    ( 0.35,  0.15,  0.0),    # forward + strafe left
    ( 0.35, -0.15,  0.0),    # forward + strafe right
    (-0.20,  0.00,  0.0),    # reverse
    ( 0.00,  0.00,  0.0),    # stop
    ( 0.40,  0.00,  0.0),    # fast forward
]


@dataclass
class GreedyConfig:
    """All hyper-parameters for the greedy energy planner."""

    # Per-dimension action bounds [vx, vy, yaw_rate]
    action_low: Tuple[float, float, float] = (-0.5, -0.3, -1.5)
    action_high: Tuple[float, float, float] = (0.5, 0.3, 1.5)

    # Scoring weights
    energy_weight: float = 1.0       # energy head cost
    beacon_weight: float = 0.5       # beacon proximity reward
    frontier_weight: float = 0.2     # frontier exploration reward

    # Contact escape parameters
    escape_reverse_steps: int = 8    # steps spent reversing
    escape_turn_steps_min: int = 10  # min steps spent turning
    escape_turn_steps_max: int = 25  # max steps spent turning
    escape_reverse_speed: float = -0.3
    escape_forward_speed: float = 0.25
    escape_yaw_min: float = 0.8
    escape_yaw_max: float = 1.4
    # Extra escape steps when collision persists during escape
    escape_extend_steps: int = 5

    # Post-escape grace period (continue in free mode without re-triggering)
    post_escape_grace: int = 8

    # CEM refinement around best primitive (0 = disabled)
    refine_candidates: int = 16
    refine_std: float = 0.08

    # Frontier grid
    frontier_cell_size: float = 0.25
    frontier_world_min: Tuple[float, float] = (-3.5, -3.5)
    frontier_world_max: Tuple[float, float] = (3.5, 3.5)

    # Simple momentum: blend previous action into scoring
    momentum: float = 0.0  # 0 = no momentum


# ---------------------------------------------------------------------- #
# Position-based frontier grid
# ---------------------------------------------------------------------- #

class FrontierGrid:
    """Grid tracking visited positions for frontier-based exploration."""

    def __init__(
        self,
        world_min: Tuple[float, float],
        world_max: Tuple[float, float],
        cell_size: float = 0.25,
    ):
        self.world_min = np.array(world_min, dtype=np.float32)
        self.world_max = np.array(world_max, dtype=np.float32)
        self.cell_size = cell_size
        extent = self.world_max - self.world_min
        self.nx = max(1, int(extent[0] / cell_size))
        self.ny = max(1, int(extent[1] / cell_size))
        self.visited = np.zeros((self.nx, self.ny), dtype=bool)

    def mark(self, x: float, y: float) -> None:
        ix = int((x - self.world_min[0]) / self.cell_size)
        iy = int((y - self.world_min[1]) / self.cell_size)
        ix = max(0, min(ix, self.nx - 1))
        iy = max(0, min(iy, self.ny - 1))
        self.visited[ix, iy] = True

    @property
    def cells_visited(self) -> int:
        return int(self.visited.sum())

    def frontier_score_for_actions(
        self,
        robot_xy: np.ndarray,
        robot_yaw: float,
        actions: np.ndarray,
        dt: float = 0.04,
    ) -> np.ndarray:
        """Score each action by how close it moves toward unvisited cells.

        Uses a simple kinematic prediction (one step) to estimate where
        each action would place the robot, then scores by inverse distance
        to the nearest unvisited cell.

        Args:
            robot_xy: (2,) current robot XY.
            robot_yaw: current heading (rad).
            actions: (N, 3) action primitives [vx, vy, yaw].
            dt: control timestep.

        Returns:
            (N,) scores, higher = more frontier-ward.
        """
        N = len(actions)
        scores = np.zeros(N, dtype=np.float32)

        # Find unvisited cells
        unvisited_ij = np.argwhere(~self.visited)
        if len(unvisited_ij) == 0:
            return scores

        # World coordinates of unvisited cell centres
        unvisited_xy = (
            self.world_min[np.newaxis, :]
            + (unvisited_ij + 0.5) * self.cell_size
        )

        cos_yaw = math.cos(robot_yaw)
        sin_yaw = math.sin(robot_yaw)

        for i in range(N):
            vx, vy, _ = float(actions[i, 0]), float(actions[i, 1]), float(actions[i, 2])
            # Predicted displacement in world frame
            dx = (vx * cos_yaw - vy * sin_yaw) * dt
            dy = (vx * sin_yaw + vy * cos_yaw) * dt
            pred_xy = robot_xy + np.array([dx, dy], dtype=np.float32)

            # Distance from predicted position to nearest unvisited cell
            dists = np.linalg.norm(unvisited_xy - pred_xy[np.newaxis, :], axis=1)
            min_dist = dists.min()
            # Inverse distance: closer to frontier = higher score
            scores[i] = 1.0 / (1.0 + min_dist)

        return scores


# ---------------------------------------------------------------------- #
# Greedy Energy Planner
# ---------------------------------------------------------------------- #

class GreedyEnergyPlanner:
    """One-step greedy planner with deterministic contact escape.

    Instead of multi-step CEM rollouts, scores discrete action primitives
    using single-step predictions through the world model.  When the
    robot is in contact with a wall (and the camera has clipped), JEPA
    observations are untrustworthy so the planner switches to a hardcoded
    reverse-and-turn escape policy.
    """

    def __init__(
        self,
        world_model: nn.Module,
        energy_head: nn.Module,
        config: GreedyConfig | None = None,
        action_set: List[Tuple[float, float, float]] | None = None,
        device: torch.device = torch.device("cpu"),
    ):
        self.wm = world_model
        self.eh = energy_head
        self.cfg = config or GreedyConfig()
        self.device = device

        primitives = action_set or DEFAULT_ACTION_SET
        self._actions = torch.tensor(primitives, dtype=torch.float32, device=device)
        self._lo = torch.tensor(self.cfg.action_low, device=device)
        self._hi = torch.tensor(self.cfg.action_high, device=device)

        # Beacon targets: {identity: z_proj tensor}
        self._beacon_targets: Dict[str, torch.Tensor] = {}
        self._captured_beacons: Set[str] = set()

        # Contact escape state
        self._escape_remaining: int = 0
        self._escape_phase: str = "reverse"  # "reverse" or "turn"
        self._escape_reverse_left: int = 0
        self._escape_yaw: float = 0.0
        self._last_escape_sign: float = 1.0
        self._grace_remaining: int = 0

        # Frontier grid
        self._frontier = FrontierGrid(
            self.cfg.frontier_world_min,
            self.cfg.frontier_world_max,
            self.cfg.frontier_cell_size,
        )

        # Tracking
        self._robot_xy: np.ndarray = np.zeros(2, dtype=np.float32)
        self._robot_yaw: float = 0.0
        self._prev_action: Optional[torch.Tensor] = None
        self._step_count: int = 0
        self._total_escapes: int = 0
        self._is_escaping: bool = False

        # Diagnostics (populated every free-mode step)
        self.diag: Dict[str, float] = {}

    # ------------------------------------------------------------------ #
    # Public interface
    # ------------------------------------------------------------------ #

    def reset(self) -> None:
        """Reset planner state for a new episode."""
        self._beacon_targets.clear()
        self._captured_beacons.clear()
        self._escape_remaining = 0
        self._grace_remaining = 0
        self._prev_action = None
        self._step_count = 0
        self._total_escapes = 0
        self._is_escaping = False
        self._last_escape_sign = 1.0
        self._frontier = FrontierGrid(
            self.cfg.frontier_world_min,
            self.cfg.frontier_world_max,
            self.cfg.frontier_cell_size,
        )
        self.diag = {}

    def set_beacon_targets(self, targets: Dict[str, torch.Tensor]) -> None:
        """Register pre-encoded beacon latents as goal targets.

        Args:
            targets: mapping from beacon identity string to projected
                     latent tensor (D,).
        """
        self._beacon_targets = {
            k: v.to(self.device) for k, v in targets.items()
        }

    def mark_captured(self, identity: str) -> None:
        """Mark a beacon as captured so it no longer attracts the robot."""
        self._captured_beacons.add(identity)

    def report_pose(self, xy: np.ndarray, yaw: float = 0.0) -> None:
        """Feed ground-truth pose for frontier scoring."""
        self._robot_xy = np.array(xy, dtype=np.float32)
        self._robot_yaw = yaw
        self._frontier.mark(float(xy[0]), float(xy[1]))

    def report_collision(self, colliding: bool) -> None:
        """Signal whether the robot is currently in physical contact."""
        # If we're in free mode and a collision is detected, enter escape
        if colliding and self._escape_remaining == 0 and self._grace_remaining == 0:
            self._start_escape(collision=True)

    @property
    def is_escaping(self) -> bool:
        return self._is_escaping

    @property
    def escape_events(self) -> int:
        return self._total_escapes

    @property
    def frontier(self) -> FrontierGrid:
        return self._frontier

    # ------------------------------------------------------------------ #
    # Step
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def step(
        self,
        vis: torch.Tensor,
        proprio: torch.Tensor | None,
        colliding: bool = False,
    ) -> torch.Tensor:
        """Plan one action.

        Args:
            vis: (1, 3, H, W) or (3, H, W) current observation.
            proprio: (1, P) or (P,) proprioception, or None.
            colliding: whether the physics engine reports wall contact.

        Returns:
            (3,) action command [vx, vy, yaw_rate].
        """
        self._step_count += 1

        # --- Contact escape mode ---
        if colliding and self._escape_remaining == 0 and self._grace_remaining == 0:
            self._start_escape(collision=True)

        if self._escape_remaining > 0:
            self._is_escaping = True
            cmd = self._escape_step(colliding)
            self._prev_action = cmd
            return cmd

        # --- Grace period after escape ---
        if self._grace_remaining > 0:
            self._grace_remaining -= 1
            self._is_escaping = False

        self._is_escaping = False

        # --- Free mode: greedy 1-step scoring ---
        if vis.dim() == 3:
            vis = vis.unsqueeze(0)
        if proprio is not None and proprio.dim() == 1:
            proprio = proprio.unsqueeze(0)

        z_raw = self.wm.encode_raw(vis, proprio)
        z_proj = self.wm.encode_observation(vis, proprio)

        cmd = self._score_actions(z_raw, z_proj)

        self._prev_action = cmd
        return cmd.clamp(self._lo, self._hi)

    # ------------------------------------------------------------------ #
    # Free-mode scoring
    # ------------------------------------------------------------------ #

    def _score_actions(
        self,
        z_raw: torch.Tensor,
        z_proj: torch.Tensor,
    ) -> torch.Tensor:
        """Score action primitives and return the best one."""
        N = self._actions.shape[0]

        # Expand z_raw for batch prediction: (N, 1, D)
        z_expand = z_raw.expand(N, -1).unsqueeze(1)
        # Actions as (N, 1, 3) — single-step sequences
        a_expand = self._actions.unsqueeze(1)

        # Single-step prediction for all primitives
        z_pred_raw = self.wm.predictor.predict_step(z_expand, a_expand)
        z_pred_proj = self.wm.pred_projector(z_pred_raw)  # (N, D)

        # Energy cost (lower energy = safer/more traversable)
        energy = self.eh(z_pred_proj)  # (N,)
        costs = self.cfg.energy_weight * energy

        # Beacon attraction
        beacon_bonus = torch.zeros(N, device=self.device)
        active_targets = {
            k: v for k, v in self._beacon_targets.items()
            if k not in self._captured_beacons
        }
        if active_targets:
            # Stack all target latents: (M, D)
            target_stack = torch.stack(list(active_targets.values()))
            # Distance from each predicted latent to each beacon: (N, M)
            dists = torch.cdist(z_pred_proj, target_stack)  # (N, M)
            # Use minimum distance to any uncaptured beacon
            min_dists = dists.min(dim=1).values  # (N,)
            # Also compute current distance for relative improvement
            cur_dists = torch.cdist(z_proj, target_stack).min(dim=1).values  # (1,)
            # Reward: how much closer does this action bring us?
            improvement = cur_dists - min_dists  # positive = good
            beacon_bonus = improvement.squeeze()
            if beacon_bonus.dim() == 0:
                beacon_bonus = beacon_bonus.unsqueeze(0)

        costs = costs - self.cfg.beacon_weight * beacon_bonus

        # Frontier exploration bonus (position-based)
        if self.cfg.frontier_weight > 0:
            frontier_scores = self._frontier.frontier_score_for_actions(
                self._robot_xy, self._robot_yaw,
                self._actions.cpu().numpy(),
            )
            frontier_t = torch.from_numpy(frontier_scores).to(self.device)
            costs = costs - self.cfg.frontier_weight * frontier_t

        # Optional momentum: slightly prefer continuing current action
        if self.cfg.momentum > 0 and self._prev_action is not None:
            similarity = (self._actions * self._prev_action.unsqueeze(0)).sum(dim=-1)
            costs = costs - self.cfg.momentum * similarity

        # Select best primitive
        best_idx = costs.argmin()
        best_action = self._actions[best_idx].clone()

        # Optional CEM refinement around the winner
        if self.cfg.refine_candidates > 0:
            best_action = self._refine(z_raw, best_action)

        # Populate diagnostics
        self.diag = {
            "cost_mean": float(costs.mean()),
            "cost_min": float(costs.min()),
            "cost_max": float(costs.max()),
            "energy_mean": float(energy.mean()),
            "energy_min": float(energy.min()),
            "best_energy": float(energy[best_idx]),
            "beacon_bonus_max": float(beacon_bonus.max()) if active_targets else 0.0,
            "best_vx": float(best_action[0]),
            "best_yaw": float(best_action[2]),
            "frontier_cells": self._frontier.cells_visited,
        }

        return best_action

    def _refine(
        self,
        z_raw: torch.Tensor,
        center: torch.Tensor,
    ) -> torch.Tensor:
        """Small CEM refinement around the best primitive."""
        K = self.cfg.refine_candidates
        noise = torch.randn(K, 3, device=self.device) * self.cfg.refine_std
        candidates = (center.unsqueeze(0) + noise).clamp(self._lo, self._hi)

        z_expand = z_raw.expand(K, -1).unsqueeze(1)
        a_expand = candidates.unsqueeze(1)

        z_pred_raw = self.wm.predictor.predict_step(z_expand, a_expand)
        z_pred_proj = self.wm.pred_projector(z_pred_raw)

        energy = self.eh(z_pred_proj)
        best_idx = energy.argmin()
        return candidates[best_idx]

    # ------------------------------------------------------------------ #
    # Contact escape
    # ------------------------------------------------------------------ #

    def _start_escape(self, collision: bool = True) -> None:
        """Initiate a reverse-then-turn escape sequence."""
        self._total_escapes += 1
        reverse_steps = self.cfg.escape_reverse_steps if collision else 0
        turn_steps = pyrandom.randint(
            self.cfg.escape_turn_steps_min, self.cfg.escape_turn_steps_max,
        )
        self._escape_remaining = reverse_steps + turn_steps
        self._escape_reverse_left = reverse_steps
        self._escape_yaw = self._last_escape_sign * pyrandom.uniform(
            self.cfg.escape_yaw_min, self.cfg.escape_yaw_max,
        )
        self._last_escape_sign *= -1.0
        self._prev_action = None

    def _escape_step(self, still_colliding: bool) -> torch.Tensor:
        """Execute one step of the escape sequence."""
        self._escape_remaining -= 1

        if self._escape_reverse_left > 0:
            # Phase 1: reverse
            self._escape_reverse_left -= 1
            cmd = torch.tensor(
                [self.cfg.escape_reverse_speed, 0.0, 0.0],
                device=self.device,
            )
            # If still colliding at end of reverse, extend
            if self._escape_reverse_left == 0 and still_colliding:
                self._escape_reverse_left += self.cfg.escape_extend_steps
                self._escape_remaining += self.cfg.escape_extend_steps
        else:
            # Phase 2: forward + turn
            cmd = torch.tensor(
                [self.cfg.escape_forward_speed, 0.0, self._escape_yaw],
                device=self.device,
            )

        if self._escape_remaining == 0:
            self._grace_remaining = self.cfg.post_escape_grace

        return cmd.clamp(self._lo, self._hi)
