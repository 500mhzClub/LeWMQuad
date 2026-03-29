"""Forward-biased explorer with geometric wall probing and JEPA beacon detection.

Lessons from v2-v7 runs:
  - JEPA 5-step rollouts can't discriminate actions (energy spread ~0.02)
  - Collision heading memory paralyses the robot (all directions penalised)
  - Beacon homing via rollout proximity doesn't work (terminal latents identical)
  - The JEPA *can* detect beacons: beacon_dist drops when beacon is in camera FOV

This version:
  - **NAVIGATE**: geometric wall probing via detect_collisions on probe points,
    picks the clearest forward direction, biased toward unvisited frontier cells.
    No JEPA rollouts. Strong forward commitment (8 steps).
  - **APPROACH**: when JEPA beacon_dist < threshold, the beacon is in the FOV.
    Walk forward with small yaw oscillation. If beacon_dist keeps dropping,
    we're approaching. If it stalls, exit back to NAVIGATE.
  - **ESCAPE**: reverse + turn on collision. No memory, no cruise.
"""
from __future__ import annotations

import math
import random as pyrandom
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Set, Tuple

import numpy as np
import torch
import torch.nn as nn


# ---------------------------------------------------------------------- #
# Configuration
# ---------------------------------------------------------------------- #

@dataclass
class ExplorerConfig:
    """All hyper-parameters."""

    action_low: Tuple[float, float, float] = (-0.5, -0.3, -1.5)
    action_high: Tuple[float, float, float] = (0.5, 0.3, 1.5)

    # --- Navigate ---
    navigate_speed: float = 0.30          # default forward speed
    navigate_hold_steps: int = 8          # commit to chosen direction
    # Geometric probing: check N directions for wall proximity
    probe_distance: float = 0.18         # how far ahead to probe (m)
    probe_n_directions: int = 13          # full 360 deg fan
    # Frontier bias: prefer unvisited cells
    frontier_weight: float = 0.8

    # --- Escape ---
    escape_reverse_steps: int = 6
    escape_turn_steps_min: int = 8
    escape_turn_steps_max: int = 18
    escape_reverse_speed: float = -0.3
    escape_turn_fwd_speed: float = 0.15
    escape_yaw_min: float = 1.0
    escape_yaw_max: float = 1.5
    escape_extend_steps: int = 5
    post_escape_grace: int = 8

    # --- Beacon approach ---
    # Entry: beacon_dist in latent space must be below this to enter approach
    approach_entry_threshold: float = 5.0
    # Exit: if beacon_dist rises above this, exit approach
    approach_exit_threshold: float = 7.0
    # Patience: max steps without improvement before exiting
    approach_patience: int = 60
    approach_min_improvement: float = 0.1
    # Approach behaviour: walk forward, oscillate yaw slightly
    approach_speed: float = 0.25
    approach_yaw_amplitude: float = 0.3   # yaw oscillation amplitude
    approach_yaw_period: int = 40         # steps per full oscillation

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

    def novelty_score(self, x: float, y: float, radius: int = 2) -> float:
        """Return fraction of cells in NxN neighbourhood that are unvisited."""
        ix = int((x - self.world_min[0]) / self.cell_size)
        iy = int((y - self.world_min[1]) / self.cell_size)
        total = 0
        unvisited = 0
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                nx_ = ix + dx
                ny_ = iy + dy
                if 0 <= nx_ < self.nx and 0 <= ny_ < self.ny:
                    total += 1
                    if not self.visited[nx_, ny_]:
                        unvisited += 1
        return unvisited / max(total, 1)


# ---------------------------------------------------------------------- #
# Main planner
# ---------------------------------------------------------------------- #

class GreedyEnergyPlanner:
    """Forward-biased geometric explorer with JEPA beacon detection.

    State machine: ESCAPE -> GRACE -> NAVIGATE <-> APPROACH
    """

    def __init__(
        self,
        world_model: nn.Module,
        energy_head: nn.Module,
        config: ExplorerConfig | None = None,
        action_set: list | None = None,  # ignored, API compat
        device: torch.device = torch.device("cpu"),
    ):
        self.wm = world_model
        self.eh = energy_head
        self.cfg = config if isinstance(config, ExplorerConfig) else ExplorerConfig()
        self.device = device

        self._lo = torch.tensor(self.cfg.action_low, device=device)
        self._hi = torch.tensor(self.cfg.action_high, device=device)

        # Beacon targets
        self._beacon_targets: Dict[str, torch.Tensor] = {}
        self._beacon_positions: Dict[str, Tuple[float, float]] = {}
        self._captured_beacons: Set[str] = set()

        # Obstacle layout for geometric probing (set by eval script)
        self._obstacle_layout = None
        self._collision_fn: Optional[Callable] = None

        # State
        self._mode: str = "navigate"
        self._step_count: int = 0
        self._total_escapes: int = 0

        # Escape
        self._escape_remaining: int = 0
        self._escape_reverse_left: int = 0
        self._escape_yaw: float = 0.0
        self._last_escape_sign: float = 1.0
        self._grace_remaining: int = 0

        # Navigate
        self._nav_cmd: Optional[torch.Tensor] = None
        self._nav_hold_remaining: int = 0

        # Approach
        self._approach_target_id: Optional[str] = None
        self._approach_best_dist: float = float("inf")
        self._approach_stale_steps: int = 0
        self._approach_step: int = 0  # for yaw oscillation

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
        self._beacon_positions.clear()
        self._captured_beacons.clear()
        self._mode = "navigate"
        self._step_count = 0
        self._total_escapes = 0
        self._escape_remaining = 0
        self._grace_remaining = 0
        self._nav_cmd = None
        self._nav_hold_remaining = 0
        self._approach_target_id = None
        self._approach_best_dist = float("inf")
        self._approach_stale_steps = 0
        self._approach_step = 0
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

    def set_beacon_positions(self, positions: Dict[str, Tuple[float, float]]) -> None:
        """Store physical XY positions of beacons for line-of-sight checks."""
        self._beacon_positions = dict(positions)

    def set_obstacle_layout(self, layout, collision_fn) -> None:
        """Provide obstacle layout and collision function for geometric probing.

        Args:
            layout: ObstacleLayout instance.
            collision_fn: callable(robot_pos_xy_tensor, layout, margin) -> bool_tensor
        """
        self._obstacle_layout = layout
        self._collision_fn = collision_fn

    def mark_captured(self, identity: str) -> None:
        self._captured_beacons.add(identity)
        if self._approach_target_id == identity:
            self._approach_target_id = None
            self._mode = "navigate"
            self._nav_hold_remaining = 0

    def report_pose(self, xy: np.ndarray, yaw: float = 0.0) -> None:
        self._robot_xy = np.array(xy, dtype=np.float32)
        self._robot_yaw = yaw
        self._frontier.mark(float(xy[0]), float(xy[1]))

    def report_collision(self, colliding: bool) -> None:
        if colliding and self._mode != "escape" and self._grace_remaining == 0:
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
            self._start_escape()

        # ESCAPE
        if self._mode == "escape":
            cmd = self._escape_step(colliding)
            self._prev_action = cmd
            return cmd

        # Grace countdown
        if self._grace_remaining > 0:
            self._grace_remaining -= 1
            if self._grace_remaining == 0:
                self._mode = "navigate"
                self._nav_hold_remaining = 0

        # JEPA beacon detection (encode once per step, lightweight)
        if vis.dim() == 3:
            vis = vis.unsqueeze(0)
        if proprio is not None and proprio.dim() == 1:
            proprio = proprio.unsqueeze(0)

        z_proj = self.wm.encode_observation(vis, proprio)
        beacon_dist, beacon_id = self._nearest_beacon_dist(z_proj)
        cur_energy = float(self.eh(z_proj).item())

        # Update diagnostics
        self.diag = {
            "mode": self._mode,
            "energy": cur_energy,
            "beacon_dist": beacon_dist if beacon_id else -1.0,
            "beacon_target": beacon_id or "none",
            "frontier_cells": self._frontier.cells_visited,
            "approach_stale": self._approach_stale_steps,
        }

        # --- Approach transitions ---
        if self._mode == "approach":
            if beacon_id and beacon_id == self._approach_target_id:
                if beacon_dist < self._approach_best_dist - self.cfg.approach_min_improvement:
                    self._approach_best_dist = beacon_dist
                    self._approach_stale_steps = 0
                else:
                    self._approach_stale_steps += 1
                # Exit conditions: stall, distance grew, or lost line-of-sight
                if (self._approach_stale_steps > self.cfg.approach_patience
                        or beacon_dist > self.cfg.approach_exit_threshold
                        or not self._has_line_of_sight(self._approach_target_id)):
                    self._mode = "navigate"
                    self._approach_target_id = None
                    self._nav_hold_remaining = 0
            else:
                self._mode = "navigate"
                self._approach_target_id = None
                self._nav_hold_remaining = 0

        elif self._mode in ("navigate", "grace"):
            if (beacon_id
                    and beacon_dist < self.cfg.approach_entry_threshold
                    and self._has_line_of_sight(beacon_id)):
                self._mode = "approach"
                self._approach_target_id = beacon_id
                self._approach_best_dist = beacon_dist
                self._approach_stale_steps = 0
                self._approach_step = 0

        # --- Execute ---
        if self._mode == "approach":
            cmd = self._approach_step_fn()
        else:
            cmd = self._navigate_step()

        self._prev_action = cmd
        return cmd.clamp(self._lo, self._hi)

    # ------------------------------------------------------------------ #
    # Navigate: geometric wall probing + frontier bias
    # ------------------------------------------------------------------ #

    def _navigate_step(self) -> torch.Tensor:
        """Pick the best forward direction using geometric collision probing."""
        # Action commitment
        if self._nav_hold_remaining > 0 and self._nav_cmd is not None:
            self._nav_hold_remaining -= 1
            return self._nav_cmd

        # Probe directions: full 360 fan (13 directions)
        # This ensures the robot can always find *some* clear direction,
        # even in tight corridors where all forward probes are blocked.
        n_dirs = self.cfg.probe_n_directions
        angles = np.linspace(-math.pi, math.pi, n_dirs, endpoint=False)

        # Probe at two distances: near (wall avoidance) and far (path finding)
        probe_near = self.cfg.probe_distance
        probe_far = self.cfg.probe_distance * 2.5

        best_yaw_offset = 0.0
        best_score = -1e9

        for angle in angles:
            world_angle = self._robot_yaw + angle

            # Near probe: is there a wall right ahead?
            near_x = self._robot_xy[0] + probe_near * math.cos(world_angle)
            near_y = self._robot_xy[1] + probe_near * math.sin(world_angle)
            near_clear = True
            if self._collision_fn is not None and self._obstacle_layout is not None:
                probe_pt = torch.tensor([[near_x, near_y]], dtype=torch.float32)
                near_clear = not bool(self._collision_fn(
                    probe_pt, self._obstacle_layout, margin=0.06,
                )[0])

            # Far probe: is the path ahead open?
            far_x = self._robot_xy[0] + probe_far * math.cos(world_angle)
            far_y = self._robot_xy[1] + probe_far * math.sin(world_angle)
            far_clear = True
            if self._collision_fn is not None and self._obstacle_layout is not None:
                probe_pt = torch.tensor([[far_x, far_y]], dtype=torch.float32)
                far_clear = not bool(self._collision_fn(
                    probe_pt, self._obstacle_layout, margin=0.06,
                )[0])

            # Score: near clearance is critical, far clearance is a bonus
            score = 0.0
            if near_clear:
                score += 2.0
                if far_clear:
                    score += 1.0
            else:
                score -= 2.0

            # Frontier novelty at the far probe point
            if self.cfg.frontier_weight > 0:
                novelty = self._frontier.novelty_score(far_x, far_y, radius=3)
                score += self.cfg.frontier_weight * novelty

            # Prefer forward (less turning) — but mild penalty so frontier
            # and clearance can override
            abs_angle = abs(angle)
            if abs_angle > math.pi:
                abs_angle = 2 * math.pi - abs_angle
            score -= 0.15 * abs_angle

            if score > best_score:
                best_score = score
                best_yaw_offset = angle

        # Wrap angle to [-pi, pi]
        if best_yaw_offset > math.pi:
            best_yaw_offset -= 2 * math.pi

        # Convert to velocity command
        yaw_rate = max(-1.5, min(1.5, best_yaw_offset * 2.0))

        # Speed: proportional to how forward the chosen direction is
        abs_yaw = abs(yaw_rate)
        if abs_yaw > 1.2:
            # Nearly turning in place — go slow
            speed = 0.08
        elif abs_yaw > 0.6:
            speed = self.cfg.navigate_speed * 0.6
        else:
            speed = self.cfg.navigate_speed

        cmd = torch.tensor([speed, 0.0, yaw_rate], device=self.device)

        # Diagnostics
        self.diag.update({
            "best_vx": float(speed),
            "best_yaw": float(yaw_rate),
            "nav_score": float(best_score),
        })

        # Commit
        self._nav_cmd = cmd
        self._nav_hold_remaining = self.cfg.navigate_hold_steps - 1

        return cmd

    # ------------------------------------------------------------------ #
    # Approach: walk toward beacon in FOV
    # ------------------------------------------------------------------ #

    def _approach_step_fn(self) -> torch.Tensor:
        """Walk forward with gentle yaw oscillation to approach visible beacon."""
        self._approach_step += 1

        # Sinusoidal yaw oscillation: helps find the beacon if it's slightly off-center
        t = self._approach_step
        period = self.cfg.approach_yaw_period
        yaw = self.cfg.approach_yaw_amplitude * math.sin(2 * math.pi * t / period)

        cmd = torch.tensor(
            [self.cfg.approach_speed, 0.0, yaw],
            device=self.device,
        )

        self.diag.update({
            "best_vx": float(cmd[0]),
            "best_yaw": float(yaw),
            "approach_dist": self._approach_best_dist,
        })

        return cmd

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
        if dists.dim() == 0:
            return float(dists), list(active.keys())[0]
        min_idx = dists.argmin()
        return float(dists[min_idx]), list(active.keys())[int(min_idx)]

    def _has_line_of_sight(self, beacon_id: str) -> bool:
        """Check if robot has clear line-of-sight to beacon (no wall between).

        Uses analytical 2D ray-AABB slab intersection for each obstacle.
        Unlike point sampling, this cannot miss thin walls.
        """
        if beacon_id not in self._beacon_positions:
            return True
        if self._obstacle_layout is None:
            return True

        bx, by = self._beacon_positions[beacon_id]
        rx, ry = float(self._robot_xy[0]), float(self._robot_xy[1])
        dx, dy = bx - rx, by - ry
        ray_len = math.sqrt(dx * dx + dy * dy)
        if ray_len < 0.05:
            return True

        for obs in self._obstacle_layout.obstacles:
            cx, cy = obs.pos[0], obs.pos[1]
            hx, hy = obs.size[0] / 2.0, obs.size[1] / 2.0
            # AABB bounds
            x_min, x_max = cx - hx, cx + hx
            y_min, y_max = cy - hy, cy + hy

            # Ray-AABB slab test (2D)
            # Ray: P(t) = (rx, ry) + t * (dx, dy), t in [0, 1]
            t_near = 0.0
            t_far = 1.0

            # X slab
            if abs(dx) < 1e-9:
                # Ray parallel to Y axis
                if rx < x_min or rx > x_max:
                    continue  # no intersection
            else:
                t1 = (x_min - rx) / dx
                t2 = (x_max - rx) / dx
                if t1 > t2:
                    t1, t2 = t2, t1
                t_near = max(t_near, t1)
                t_far = min(t_far, t2)
                if t_near > t_far:
                    continue

            # Y slab
            if abs(dy) < 1e-9:
                if ry < y_min or ry > y_max:
                    continue
            else:
                t1 = (y_min - ry) / dy
                t2 = (y_max - ry) / dy
                if t1 > t2:
                    t1, t2 = t2, t1
                t_near = max(t_near, t1)
                t_far = min(t_far, t2)
                if t_near > t_far:
                    continue

            # Ray intersects this obstacle AABB in [t_near, t_far] ⊂ [0, 1]
            # Skip intersection if it's only at the very end (beacon is ON a wall)
            if t_near < 0.95:
                return False

        return True

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
        self._nav_cmd = None
        self._nav_hold_remaining = 0
        self._approach_target_id = None

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
