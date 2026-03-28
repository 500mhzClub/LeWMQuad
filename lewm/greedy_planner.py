"""Energy-scored explorer with collision heading memory and beacon homing.

Hybrid approach: energy-scored action selection (works in open corridors)
combined with collision heading memory (avoids returning to the wall just
escaped from) and JEPA-guided beacon homing when close.

State machine:
  - **ESCAPE**: reverse + ~90 deg turn.  Records the heading that caused
    collision.  On completion -> GRACE -> NAVIGATE.
  - **NAVIGATE**: energy-scored action selection with collision heading
    penalty.  Short action commitment (4-6 steps).  Checks beacon
    proximity each cycle and switches to BEACON_HOME if close.
  - **BEACON_HOME**: JEPA rollout scores actions by beacon proximity
    improvement.  Drops back to NAVIGATE on stall or collision.

Key difference from v2: no cruise mode (caused ESCAPE->CRUISE->ESCAPE
trap).  Instead, collision heading memory biases the robot away from
the wall it just left without committing to a blind forward drive.
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
ACTION_SET: List[Tuple[float, float, float]] = [
    ( 0.35,  0.00,  0.0),    # forward
    ( 0.30,  0.00,  0.3),    # forward-left
    ( 0.30,  0.00, -0.3),    # forward-right
    ( 0.20,  0.00,  0.6),    # left
    ( 0.20,  0.00, -0.6),    # right
    ( 0.10,  0.00,  1.0),    # sharp left
    ( 0.10,  0.00, -1.0),    # sharp right
    ( 0.35,  0.12,  0.0),    # forward + strafe left
    ( 0.35, -0.12,  0.0),    # forward + strafe right
    (-0.15,  0.00,  0.0),    # slow reverse
    ( 0.00,  0.00,  1.2),    # spin left
    ( 0.00,  0.00, -1.2),    # spin right
]


@dataclass
class ExplorerConfig:
    """All hyper-parameters for the energy-scored explorer."""

    # Per-dimension action bounds
    action_low: Tuple[float, float, float] = (-0.5, -0.3, -1.5)
    action_high: Tuple[float, float, float] = (0.5, 0.3, 1.5)

    # --- Navigation scoring ---
    hold_steps: int = 5              # rollout length for scoring
    energy_weight: float = 0.3       # low — energy head is noisy near walls
    forward_bonus: float = 0.08      # adaptive, scales with energy spread
    beacon_weight: float = 0.5       # beacon proximity in latent space
    collision_heading_weight: float = 0.4  # penalty for facing recent collision headings
    collision_heading_memory: int = 5      # remember last N collision headings
    collision_heading_decay: float = 0.8   # older collisions matter less

    # Action commitment: hold chosen action for N steps
    action_hold_steps: int = 5

    # --- Escape ---
    escape_reverse_steps: int = 8
    escape_turn_steps_min: int = 12
    escape_turn_steps_max: int = 25
    escape_reverse_speed: float = -0.3
    escape_turn_fwd_speed: float = 0.15
    escape_yaw_min: float = 1.0
    escape_yaw_max: float = 1.5
    escape_extend_steps: int = 5

    # Grace period after escape
    post_escape_grace: int = 10

    # --- Beacon homing ---
    homing_hold_steps: int = 5
    homing_patience: int = 40
    homing_entry_threshold: float = 8.0
    homing_min_improvement: float = 0.05
    homing_action_hold: int = 5

    # --- Frontier grid ---
    frontier_cell_size: float = 0.25
    frontier_world_min: Tuple[float, float] = (-3.5, -3.5)
    frontier_world_max: Tuple[float, float] = (3.5, 3.5)


# ---------------------------------------------------------------------- #
# Frontier grid
# ---------------------------------------------------------------------- #

class FrontierGrid:
    """Grid tracking visited positions."""

    def __init__(self, world_min, world_max, cell_size=0.25):
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


# ---------------------------------------------------------------------- #
# Main planner
# ---------------------------------------------------------------------- #

class GreedyEnergyPlanner:
    """Energy-scored explorer with collision memory and beacon homing.

    State machine: ESCAPE -> GRACE -> NAVIGATE <-> BEACON_HOME
    """

    def __init__(
        self,
        world_model: nn.Module,
        energy_head: nn.Module,
        config: ExplorerConfig | None = None,
        action_set: list | None = None,
        device: torch.device = torch.device("cpu"),
    ):
        self.wm = world_model
        self.eh = energy_head
        self.cfg = config if isinstance(config, ExplorerConfig) else ExplorerConfig()
        self.device = device

        primitives = action_set or ACTION_SET
        self._actions = torch.tensor(primitives, dtype=torch.float32, device=device)
        self._lo = torch.tensor(self.cfg.action_low, device=device)
        self._hi = torch.tensor(self.cfg.action_high, device=device)

        # Beacon targets
        self._beacon_targets: Dict[str, torch.Tensor] = {}
        self._captured_beacons: Set[str] = set()

        # State
        self._mode: str = "navigate"
        self._step_count: int = 0
        self._total_escapes: int = 0

        # Escape state
        self._escape_remaining: int = 0
        self._escape_reverse_left: int = 0
        self._escape_yaw: float = 0.0
        self._last_escape_sign: float = 1.0

        # Grace state
        self._grace_remaining: int = 0

        # Action commitment
        self._held_action: Optional[torch.Tensor] = None
        self._hold_remaining: int = 0

        # Collision heading memory: deque of (yaw_at_collision,)
        self._collision_headings: deque = deque(
            maxlen=self.cfg.collision_heading_memory,
        )

        # Beacon homing state
        self._homing_target_id: Optional[str] = None
        self._homing_best_dist: float = float("inf")
        self._homing_stale_steps: int = 0
        self._homing_held_action: Optional[torch.Tensor] = None
        self._homing_hold_remaining: int = 0

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

        # Diagnostics
        self.diag: Dict[str, float] = {}

    # ------------------------------------------------------------------ #
    # Public interface
    # ------------------------------------------------------------------ #

    def reset(self) -> None:
        self._beacon_targets.clear()
        self._captured_beacons.clear()
        self._mode = "navigate"
        self._step_count = 0
        self._total_escapes = 0
        self._escape_remaining = 0
        self._grace_remaining = 0
        self._held_action = None
        self._hold_remaining = 0
        self._collision_headings.clear()
        self._homing_target_id = None
        self._homing_best_dist = float("inf")
        self._homing_stale_steps = 0
        self._homing_held_action = None
        self._homing_hold_remaining = 0
        self._last_escape_sign = 1.0
        self._prev_action = None
        self._frontier = FrontierGrid(
            self.cfg.frontier_world_min,
            self.cfg.frontier_world_max,
            self.cfg.frontier_cell_size,
        )
        self.diag = {}

    def set_beacon_targets(self, targets: Dict[str, torch.Tensor]) -> None:
        self._beacon_targets = {k: v.to(self.device) for k, v in targets.items()}

    def mark_captured(self, identity: str) -> None:
        self._captured_beacons.add(identity)
        if self._homing_target_id == identity:
            self._homing_target_id = None
            self._mode = "navigate"
            self._hold_remaining = 0

    def report_pose(self, xy: np.ndarray, yaw: float = 0.0) -> None:
        self._robot_xy = np.array(xy, dtype=np.float32)
        self._robot_yaw = yaw
        self._frontier.mark(float(xy[0]), float(xy[1]))

    def report_collision(self, colliding: bool) -> None:
        if colliding and self._mode != "escape" and self._grace_remaining == 0:
            self._collision_headings.append(self._robot_yaw)
            self._start_escape()

    @property
    def is_escaping(self) -> bool:
        return self._mode == "escape"

    @property
    def is_cruising(self) -> bool:
        return False

    @property
    def escape_events(self) -> int:
        return self._total_escapes

    @property
    def frontier(self) -> FrontierGrid:
        return self._frontier

    @property
    def mode(self) -> str:
        return self._mode

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
        self._step_count += 1

        # Collision -> escape
        if colliding and self._mode != "escape" and self._grace_remaining == 0:
            self._collision_headings.append(self._robot_yaw)
            self._start_escape()

        # ESCAPE mode
        if self._mode == "escape":
            cmd = self._escape_step(colliding)
            self._prev_action = cmd
            return cmd

        # Grace countdown
        if self._grace_remaining > 0:
            self._grace_remaining -= 1
            if self._grace_remaining == 0:
                self._mode = "navigate"
                self._hold_remaining = 0

        # Prep observation tensors
        if vis.dim() == 3:
            vis = vis.unsqueeze(0)
        if proprio is not None and proprio.dim() == 1:
            proprio = proprio.unsqueeze(0)

        z_proj = self.wm.encode_observation(vis, proprio)
        z_raw = self.wm.encode_raw(vis, proprio)

        # Beacon proximity check
        beacon_dist, beacon_id = self._nearest_beacon_dist(z_proj)
        cur_energy = float(self.eh(z_proj).item())

        # Update diagnostics
        self.diag = {
            "mode": self._mode,
            "energy": cur_energy,
            "beacon_dist": beacon_dist if beacon_id else -1.0,
            "beacon_target": beacon_id or "none",
            "frontier_cells": self._frontier.cells_visited,
            "homing_stale": self._homing_stale_steps,
            "n_collision_headings": len(self._collision_headings),
        }

        # --- Beacon homing transitions ---
        if self._mode == "beacon_home":
            if beacon_id and beacon_id == self._homing_target_id:
                if beacon_dist < self._homing_best_dist - self.cfg.homing_min_improvement:
                    self._homing_best_dist = beacon_dist
                    self._homing_stale_steps = 0
                else:
                    self._homing_stale_steps += 1
                if self._homing_stale_steps > self.cfg.homing_patience:
                    self._mode = "navigate"
                    self._homing_target_id = None
                    self._hold_remaining = 0
            else:
                self._mode = "navigate"
                self._homing_target_id = None
                self._hold_remaining = 0

        elif self._mode in ("navigate", "grace"):
            if beacon_id and beacon_dist < self.cfg.homing_entry_threshold:
                self._mode = "beacon_home"
                self._homing_target_id = beacon_id
                self._homing_best_dist = beacon_dist
                self._homing_stale_steps = 0
                self._homing_held_action = None
                self._homing_hold_remaining = 0

        # --- Execute ---
        if self._mode == "beacon_home":
            cmd = self._beacon_homing_step(z_raw, z_proj)
        else:
            cmd = self._navigate_step(z_raw, z_proj)

        self._prev_action = cmd
        return cmd.clamp(self._lo, self._hi)

    # ------------------------------------------------------------------ #
    # Navigate: energy-scored with collision heading memory
    # ------------------------------------------------------------------ #

    def _navigate_step(
        self,
        z_raw: torch.Tensor,
        z_proj: torch.Tensor,
    ) -> torch.Tensor:
        """Score action primitives using energy + collision memory + forward bonus."""
        # Action commitment
        if self._hold_remaining > 0 and self._held_action is not None:
            self._hold_remaining -= 1
            return self._held_action

        N = self._actions.shape[0]
        H = self.cfg.hold_steps

        z_start = z_raw.expand(N, -1)
        action_seq = self._actions.unsqueeze(1).expand(N, H, 3)

        # Rollout
        z_pred_seq = self.wm.plan_rollout(z_start, action_seq)  # (N, H, D)

        # Energy score (low weight — noisy near walls but useful in corridors)
        energy = self.eh.score_trajectory(z_pred_seq) / float(H)  # (N,)
        costs = self.cfg.energy_weight * energy

        # Adaptive forward bonus
        e_spread = float(energy.max() - energy.min())
        if self.cfg.forward_bonus > 0:
            adaptive_scale = self.cfg.forward_bonus * (1.0 + e_spread)
            costs = costs - adaptive_scale * self._actions[:, 0]

        # Collision heading penalty: penalize actions that move toward
        # recently-collided headings.  Each action's "world heading" is
        # robot_yaw + atan2(vy, vx) for the action.  We penalize by
        # angular proximity to each remembered collision heading, with
        # exponential decay for older collisions.
        if self._collision_headings and self.cfg.collision_heading_weight > 0:
            penalty = torch.zeros(N, device=self.device)
            n_mem = len(self._collision_headings)
            for i, col_yaw in enumerate(self._collision_headings):
                # Newer collisions have higher weight
                age_weight = self.cfg.collision_heading_decay ** (n_mem - 1 - i)
                for j in range(N):
                    vx = float(self._actions[j, 0])
                    vy = float(self._actions[j, 1])
                    yaw_rate = float(self._actions[j, 2])
                    # Predicted heading after this action
                    if abs(vx) > 0.01 or abs(vy) > 0.01:
                        action_heading = self._robot_yaw + math.atan2(vy, vx)
                    else:
                        # Pure rotation: heading after yaw_rate * dt * hold_steps
                        action_heading = self._robot_yaw + yaw_rate * 0.04 * H
                    # Angular distance to collision heading
                    delta = action_heading - col_yaw
                    delta = math.atan2(math.sin(delta), math.cos(delta))
                    # Gaussian-ish penalty: strong when facing collision direction
                    penalty[j] += age_weight * math.exp(-delta * delta / 0.5)

            costs = costs + self.cfg.collision_heading_weight * penalty

        # Beacon attraction (even in navigate — gentle pull toward beacons)
        z_terminal = z_pred_seq[:, -1, :]
        active_targets = {
            k: v for k, v in self._beacon_targets.items()
            if k not in self._captured_beacons
        }
        beacon_bonus = torch.zeros(N, device=self.device)
        if active_targets:
            target_stack = torch.stack(list(active_targets.values()))
            dists = torch.cdist(z_terminal, target_stack)
            min_dists = dists.min(dim=1).values
            cur_dists = torch.cdist(z_proj, target_stack).min(dim=1).values
            improvement = cur_dists - min_dists
            beacon_bonus = improvement.squeeze()
            if beacon_bonus.dim() == 0:
                beacon_bonus = beacon_bonus.unsqueeze(0)
        costs = costs - self.cfg.beacon_weight * beacon_bonus

        # Select best
        best_idx = costs.argmin()
        best_action = self._actions[best_idx].clone()

        # Diagnostics
        self.diag.update({
            "cost_min": float(costs.min()),
            "cost_mean": float(costs.mean()),
            "energy_mean": float(energy.mean()),
            "energy_spread": e_spread,
            "best_vx": float(best_action[0]),
            "best_yaw": float(best_action[2]),
            "beacon_bonus_max": float(beacon_bonus.max()) if active_targets else 0.0,
        })

        # Commit
        self._held_action = best_action
        self._hold_remaining = self.cfg.action_hold_steps - 1

        return best_action

    # ------------------------------------------------------------------ #
    # Beacon homing
    # ------------------------------------------------------------------ #

    def _beacon_homing_step(
        self,
        z_raw: torch.Tensor,
        z_proj: torch.Tensor,
    ) -> torch.Tensor:
        """Score actions purely by beacon proximity improvement."""
        if self._homing_hold_remaining > 0 and self._homing_held_action is not None:
            self._homing_hold_remaining -= 1
            return self._homing_held_action

        target_z = self._beacon_targets.get(self._homing_target_id)
        if target_z is None:
            self._mode = "navigate"
            return self._navigate_step(z_raw, z_proj)

        N = self._actions.shape[0]
        H = self.cfg.homing_hold_steps

        z_start = z_raw.expand(N, -1)
        action_seq = self._actions.unsqueeze(1).expand(N, H, 3)
        z_pred_seq = self.wm.plan_rollout(z_start, action_seq)
        z_terminal = z_pred_seq[:, -1, :]

        target_expanded = target_z.unsqueeze(0).expand(N, -1)
        dists = torch.norm(z_terminal - target_expanded, dim=-1)

        cur_dist = torch.norm(z_proj - target_z.unsqueeze(0), dim=-1)

        best_idx = dists.argmin()
        best_action = self._actions[best_idx].clone()

        self.diag.update({
            "homing_cur_dist": float(cur_dist),
            "homing_pred_best": float(dists[best_idx]),
            "best_vx": float(best_action[0]),
            "best_yaw": float(best_action[2]),
        })

        self._homing_held_action = best_action
        self._homing_hold_remaining = self.cfg.homing_action_hold - 1
        return best_action

    # ------------------------------------------------------------------ #
    # Beacon proximity
    # ------------------------------------------------------------------ #

    def _nearest_beacon_dist(self, z_proj: torch.Tensor) -> Tuple[float, Optional[str]]:
        active = {
            k: v for k, v in self._beacon_targets.items()
            if k not in self._captured_beacons
        }
        if not active:
            return float("inf"), None
        target_stack = torch.stack(list(active.values()))
        dists = torch.cdist(z_proj, target_stack).squeeze(0)
        min_idx = dists.argmin()
        return float(dists[min_idx]), list(active.keys())[int(min_idx)]

    # ------------------------------------------------------------------ #
    # Escape
    # ------------------------------------------------------------------ #

    def _start_escape(self) -> None:
        self._total_escapes += 1
        self._mode = "escape"
        reverse_steps = self.cfg.escape_reverse_steps
        turn_steps = pyrandom.randint(
            self.cfg.escape_turn_steps_min, self.cfg.escape_turn_steps_max,
        )
        self._escape_remaining = reverse_steps + turn_steps
        self._escape_reverse_left = reverse_steps
        self._escape_yaw = self._last_escape_sign * pyrandom.uniform(
            self.cfg.escape_yaw_min, self.cfg.escape_yaw_max,
        )
        self._last_escape_sign *= -1.0
        self._held_action = None
        self._hold_remaining = 0
        self._homing_held_action = None
        self._homing_hold_remaining = 0
        self._homing_target_id = None

    def _escape_step(self, still_colliding: bool) -> torch.Tensor:
        self._escape_remaining -= 1
        if self._escape_reverse_left > 0:
            self._escape_reverse_left -= 1
            cmd = torch.tensor(
                [self.cfg.escape_reverse_speed, 0.0, 0.0],
                device=self.device,
            )
            if self._escape_reverse_left == 0 and still_colliding:
                self._escape_reverse_left += self.cfg.escape_extend_steps
                self._escape_remaining += self.cfg.escape_extend_steps
        else:
            cmd = torch.tensor(
                [self.cfg.escape_turn_fwd_speed, 0.0, self._escape_yaw],
                device=self.device,
            )
        if self._escape_remaining == 0:
            self._mode = "grace"
            self._grace_remaining = self.cfg.post_escape_grace
        return cmd.clamp(self._lo, self._hi)
