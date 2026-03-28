"""Random-walk explorer with JEPA beacon homing.

The energy head cannot see walls (trained on camera-clipped data), so
we abandon energy-based navigation entirely for exploration.  Instead:

  - **WANDER**: random forward-biased walk.  Pick a random action
    (weighted toward forward), hold for 15-25 steps, repeat.  On
    collision -> ESCAPE.
  - **ESCAPE**: deterministic reverse + large turn (~90 deg).  On
    completion -> GRACE -> WANDER with a new random heading.
  - **BEACON_HOME**: when the current observation is "close" to a
    beacon latent (and getting closer), switch to JEPA-guided action
    selection that scores primitives purely by beacon proximity
    improvement.  On stall or collision -> back to WANDER.

Typical usage::

    planner = RandomExplorerPlanner(world_model, energy_head, device=device)
    planner.set_beacon_targets(beacon_latents)

    for obs in environment:
        action = planner.step(obs.vision, obs.proprio, obs.colliding)
        env.apply(action)
    planner.reset()
"""
from __future__ import annotations

import math
import random as pyrandom
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import torch
import torch.nn as nn


# ---------------------------------------------------------------------- #
# Configuration
# ---------------------------------------------------------------------- #

# Discrete action primitives for beacon homing: (vx, vy, yaw_rate)
HOMING_ACTION_SET: List[Tuple[float, float, float]] = [
    ( 0.35,  0.00,  0.0),    # forward
    ( 0.30,  0.00,  0.3),    # forward-left
    ( 0.30,  0.00, -0.3),    # forward-right
    ( 0.20,  0.00,  0.6),    # left
    ( 0.20,  0.00, -0.6),    # right
    ( 0.10,  0.00,  1.0),    # sharp left
    ( 0.10,  0.00, -1.0),    # sharp right
    ( 0.35,  0.12,  0.0),    # forward + strafe left
    ( 0.35, -0.12,  0.0),    # forward + strafe right
]

# Random wander actions: forward-biased with occasional turns
WANDER_ACTION_SET: List[Tuple[float, float, float]] = [
    # Forward variants (high probability weight)
    ( 0.35,  0.00,  0.0),    # straight
    ( 0.35,  0.00,  0.0),    # straight (duplicated for bias)
    ( 0.35,  0.00,  0.0),    # straight (duplicated for bias)
    ( 0.30,  0.00,  0.15),   # gentle left
    ( 0.30,  0.00, -0.15),   # gentle right
    ( 0.25,  0.00,  0.35),   # moderate left
    ( 0.25,  0.00, -0.35),   # moderate right
    # Turn variants (lower probability)
    ( 0.15,  0.00,  0.7),    # hard left
    ( 0.15,  0.00, -0.7),    # hard right
]


@dataclass
class ExplorerConfig:
    """All hyper-parameters for the random-walk explorer."""

    # Per-dimension action bounds [vx, vy, yaw_rate]
    action_low: Tuple[float, float, float] = (-0.5, -0.3, -1.5)
    action_high: Tuple[float, float, float] = (0.5, 0.3, 1.5)

    # --- Wander ---
    wander_hold_min: int = 15        # min steps to hold each random action
    wander_hold_max: int = 30        # max steps

    # --- Escape ---
    escape_reverse_steps: int = 10   # steps spent reversing
    escape_turn_steps_min: int = 15  # min turn steps (~90 deg at yaw=1.2)
    escape_turn_steps_max: int = 30  # max turn steps
    escape_reverse_speed: float = -0.3
    escape_turn_fwd_speed: float = 0.15   # slight forward during turn
    escape_yaw_min: float = 1.0      # ~57 deg/sec
    escape_yaw_max: float = 1.5      # ~86 deg/sec
    escape_extend_steps: int = 5     # extend reverse if still colliding

    # Grace period after escape before collision detection re-arms
    post_escape_grace: int = 12

    # --- Beacon homing ---
    # Rollout length for scoring beacon homing actions
    homing_hold_steps: int = 5
    # How many steps to stay in homing mode without improvement before dropping back
    homing_patience: int = 30
    # Beacon proximity threshold to enter homing (L2 distance in projected latent space)
    # Lower = stricter, only home when very close.  Start generous and tune down.
    homing_entry_threshold: float = 8.0
    # Minimum improvement per homing_patience window to stay in homing
    homing_min_improvement: float = 0.05
    # Action commitment during homing
    homing_action_hold: int = 5

    # --- Frontier grid ---
    frontier_cell_size: float = 0.25
    frontier_world_min: Tuple[float, float] = (-3.5, -3.5)
    frontier_world_max: Tuple[float, float] = (3.5, 3.5)


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


# ---------------------------------------------------------------------- #
# Random Explorer with Beacon Homing
# ---------------------------------------------------------------------- #

class GreedyEnergyPlanner:
    """Random-walk explorer with JEPA-guided beacon homing.

    State machine: ESCAPE -> GRACE -> WANDER <-> BEACON_HOME
    """

    def __init__(
        self,
        world_model: nn.Module,
        energy_head: nn.Module,
        config: ExplorerConfig | None = None,
        action_set: list | None = None,    # ignored, kept for API compat
        device: torch.device = torch.device("cpu"),
    ):
        self.wm = world_model
        self.eh = energy_head  # kept for diagnostics only
        self.cfg = config if isinstance(config, ExplorerConfig) else ExplorerConfig()
        self.device = device

        self._homing_actions = torch.tensor(
            HOMING_ACTION_SET, dtype=torch.float32, device=device,
        )
        self._lo = torch.tensor(self.cfg.action_low, device=device)
        self._hi = torch.tensor(self.cfg.action_high, device=device)

        # Beacon targets: {identity: z_proj tensor}
        self._beacon_targets: Dict[str, torch.Tensor] = {}
        self._captured_beacons: Set[str] = set()

        # State
        self._mode: str = "wander"  # "wander", "escape", "grace", "beacon_home"
        self._step_count: int = 0
        self._total_escapes: int = 0

        # Escape state
        self._escape_remaining: int = 0
        self._escape_phase: str = "reverse"
        self._escape_reverse_left: int = 0
        self._escape_yaw: float = 0.0
        self._last_escape_sign: float = 1.0

        # Grace state
        self._grace_remaining: int = 0

        # Wander state
        self._wander_cmd: Optional[torch.Tensor] = None
        self._wander_remaining: int = 0

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
        self._mode = "wander"
        self._step_count = 0
        self._total_escapes = 0
        self._escape_remaining = 0
        self._grace_remaining = 0
        self._wander_cmd = None
        self._wander_remaining = 0
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
        self._beacon_targets = {
            k: v.to(self.device) for k, v in targets.items()
        }

    def mark_captured(self, identity: str) -> None:
        self._captured_beacons.add(identity)
        if self._homing_target_id == identity:
            self._homing_target_id = None
            self._mode = "wander"
            self._wander_remaining = 0  # pick new wander action immediately

    def report_pose(self, xy: np.ndarray, yaw: float = 0.0) -> None:
        self._robot_xy = np.array(xy, dtype=np.float32)
        self._robot_yaw = yaw
        self._frontier.mark(float(xy[0]), float(xy[1]))

    def report_collision(self, colliding: bool) -> None:
        """Signal collision from physics. Triggers escape if not already escaping."""
        if colliding and self._mode not in ("escape",) and self._grace_remaining == 0:
            self._start_escape()

    @property
    def is_escaping(self) -> bool:
        return self._mode == "escape"

    @property
    def is_cruising(self) -> bool:
        return False  # no cruise mode

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
        """Plan one action.

        Args:
            vis: (1, 3, H, W) or (3, H, W) current observation.
            proprio: (1, P) or (P,) proprioception, or None.
            colliding: whether the physics engine reports wall contact.

        Returns:
            (3,) action command [vx, vy, yaw_rate].
        """
        self._step_count += 1

        # --- Collision -> escape (unless already escaping or in grace) ---
        if colliding and self._mode != "escape" and self._grace_remaining == 0:
            self._start_escape()

        # --- ESCAPE mode ---
        if self._mode == "escape":
            cmd = self._escape_step(colliding)
            self._prev_action = cmd
            return cmd

        # --- GRACE period ---
        if self._grace_remaining > 0:
            self._grace_remaining -= 1
            if self._grace_remaining == 0:
                self._mode = "wander"
                self._wander_remaining = 0  # pick fresh action

        # --- Encode observation (needed for beacon check & homing) ---
        if vis.dim() == 3:
            vis = vis.unsqueeze(0)
        if proprio is not None and proprio.dim() == 1:
            proprio = proprio.unsqueeze(0)

        z_proj = self.wm.encode_observation(vis, proprio)
        z_raw = self.wm.encode_raw(vis, proprio)

        # --- Check beacon proximity -> maybe switch to homing ---
        beacon_dist, beacon_id = self._nearest_beacon_dist(z_proj)
        cur_energy = float(self.eh(z_proj).item())

        # Update diagnostics
        self.diag = {
            "mode": self._mode,
            "energy": cur_energy,
            "beacon_dist": beacon_dist if beacon_id else -1.0,
            "beacon_target": beacon_id or "none",
            "frontier_cells": self._frontier.cells_visited,
            "wander_remaining": self._wander_remaining,
            "homing_stale": self._homing_stale_steps,
        }

        # Decide whether to enter/stay in beacon homing
        if self._mode == "beacon_home":
            # Already homing — check if we should continue
            if beacon_id and beacon_id == self._homing_target_id:
                if beacon_dist < self._homing_best_dist - self.cfg.homing_min_improvement:
                    # Making progress
                    self._homing_best_dist = beacon_dist
                    self._homing_stale_steps = 0
                else:
                    self._homing_stale_steps += 1

                if self._homing_stale_steps > self.cfg.homing_patience:
                    # Stalled, drop back to wander
                    self._mode = "wander"
                    self._wander_remaining = 0
                    self._homing_target_id = None
            else:
                # Target disappeared or was captured
                self._mode = "wander"
                self._wander_remaining = 0
                self._homing_target_id = None

        elif self._mode in ("wander", "grace"):
            # Check if we should enter homing
            if beacon_id and beacon_dist < self.cfg.homing_entry_threshold:
                self._mode = "beacon_home"
                self._homing_target_id = beacon_id
                self._homing_best_dist = beacon_dist
                self._homing_stale_steps = 0
                self._homing_held_action = None
                self._homing_hold_remaining = 0

        # --- Execute current mode ---
        if self._mode == "beacon_home":
            cmd = self._beacon_homing_step(z_raw, z_proj)
        else:
            cmd = self._wander_step()

        self._prev_action = cmd
        return cmd.clamp(self._lo, self._hi)

    # ------------------------------------------------------------------ #
    # Wander mode
    # ------------------------------------------------------------------ #

    def _wander_step(self) -> torch.Tensor:
        """Pick/hold a random forward-biased action."""
        if self._wander_remaining <= 0 or self._wander_cmd is None:
            # Pick a new random action from the wander set
            idx = pyrandom.randrange(len(WANDER_ACTION_SET))
            action = WANDER_ACTION_SET[idx]
            self._wander_cmd = torch.tensor(action, dtype=torch.float32, device=self.device)
            self._wander_remaining = pyrandom.randint(
                self.cfg.wander_hold_min, self.cfg.wander_hold_max,
            )

        self._wander_remaining -= 1
        return self._wander_cmd

    # ------------------------------------------------------------------ #
    # Beacon homing mode
    # ------------------------------------------------------------------ #

    def _beacon_homing_step(
        self,
        z_raw: torch.Tensor,
        z_proj: torch.Tensor,
    ) -> torch.Tensor:
        """Score homing action primitives by beacon proximity improvement."""
        # Action commitment
        if self._homing_hold_remaining > 0 and self._homing_held_action is not None:
            self._homing_hold_remaining -= 1
            return self._homing_held_action

        target_z = self._beacon_targets.get(self._homing_target_id)
        if target_z is None:
            # Target gone, fall back
            self._mode = "wander"
            return self._wander_step()

        N = self._homing_actions.shape[0]
        H = self.cfg.homing_hold_steps

        z_start = z_raw.expand(N, -1)
        action_seq = self._homing_actions.unsqueeze(1).expand(N, H, 3)

        # Rollout predictor (returns projected latents)
        z_pred_seq = self.wm.plan_rollout(z_start, action_seq)  # (N, H, D_proj)
        z_terminal = z_pred_seq[:, -1, :]  # (N, D_proj)

        # Score by distance to target beacon in projected space
        target_expanded = target_z.unsqueeze(0).expand(N, -1)
        dists = torch.norm(z_terminal - target_expanded, dim=-1)  # (N,)

        # Current distance for reference
        cur_dist = torch.norm(z_proj - target_z.unsqueeze(0), dim=-1)  # (1,)

        # Best action = smallest predicted distance to beacon
        best_idx = dists.argmin()
        best_action = self._homing_actions[best_idx].clone()

        # Update diagnostics
        self.diag.update({
            "homing_cur_dist": float(cur_dist),
            "homing_pred_best": float(dists[best_idx]),
            "homing_pred_worst": float(dists.max()),
            "best_vx": float(best_action[0]),
            "best_yaw": float(best_action[2]),
        })

        # Commit
        self._homing_held_action = best_action
        self._homing_hold_remaining = self.cfg.homing_action_hold - 1

        return best_action

    # ------------------------------------------------------------------ #
    # Beacon proximity check
    # ------------------------------------------------------------------ #

    def _nearest_beacon_dist(
        self,
        z_proj: torch.Tensor,
    ) -> Tuple[float, Optional[str]]:
        """Return (distance, identity) of the nearest uncaptured beacon."""
        active = {
            k: v for k, v in self._beacon_targets.items()
            if k not in self._captured_beacons
        }
        if not active:
            return float("inf"), None

        target_stack = torch.stack(list(active.values()))  # (M, D)
        dists = torch.cdist(z_proj, target_stack).squeeze(0)  # (M,)
        min_idx = dists.argmin()
        min_dist = float(dists[min_idx])
        identity = list(active.keys())[int(min_idx)]
        return min_dist, identity

    # ------------------------------------------------------------------ #
    # Contact escape
    # ------------------------------------------------------------------ #

    def _start_escape(self) -> None:
        """Initiate a reverse-then-turn escape sequence."""
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

        # Kill any held actions
        self._wander_cmd = None
        self._wander_remaining = 0
        self._homing_held_action = None
        self._homing_hold_remaining = 0
        self._homing_target_id = None

    def _escape_step(self, still_colliding: bool) -> torch.Tensor:
        """Execute one step of the escape sequence."""
        self._escape_remaining -= 1

        if self._escape_reverse_left > 0:
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
            # Turn phase: slight forward + yaw
            cmd = torch.tensor(
                [self.cfg.escape_turn_fwd_speed, 0.0, self._escape_yaw],
                device=self.device,
            )

        if self._escape_remaining == 0:
            # Transition to grace -> wander
            self._mode = "grace"
            self._grace_remaining = self.cfg.post_escape_grace

        return cmd.clamp(self._lo, self._hi)
