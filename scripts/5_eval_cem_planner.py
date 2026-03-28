#!/usr/bin/env python3
"""Evaluate the CEM planner with energy head in the Genesis simulator.

Three-camera setup matching TinyQuadJEPA-v2 visualization:
  - cam_brain (224×224, FOV 58°) — egocentric for JEPA encoder input
  - cam_eye   (384×384, FOV 58°) — high-res first-person for display
  - cam_over  (512×512, FOV 55°) — third-person chase-cam

Composed HUD (896×660) with beacon pills, status text, minimap, event log.

Usage:
    python scripts/5_eval_cem_planner.py \\
        --ppo_ckpt assets/ppo_walking.pt \\
        --lewm_ckpt lewm_checkpoints/epoch_20.pt \\
        --energy_ckpt energy_head_checkpoints/energy_epoch_10.pt \\
        --max_steps 2000 --seed 42
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections import deque
from typing import Deque, List, Optional, Tuple

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import imageio
import numpy as np
import torch
from PIL import Image, ImageDraw

from lewm.models import LeWorldModel, LatentEnergyHead, ActorCritic
from lewm.planner import CEMPlanner, CEMConfig
from lewm.checkpoint_utils import clean_state_dict, load_ppo_checkpoint
from lewm.math_utils import forward_up_from_quat, world_to_body_vec
from lewm.genesis_utils import init_genesis_once, to_numpy
from lewm.obstacle_utils import (
    ObstacleLayout,
    add_obstacles_to_scene,
    detect_collisions,
)
from lewm.maze_utils import generate_composite_scene, MAZE_STYLES
from lewm.beacon_utils import BeaconLayout, BEACON_FAMILIES

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

URDF_PATH = "assets/mini_pupper/mini_pupper.urdf"
JOINTS_ACTUATED = [
    "lf_hip_joint",  "lh_hip_joint",  "rf_hip_joint",  "rh_hip_joint",
    "lf_thigh_joint","lh_thigh_joint","rf_thigh_joint","rh_thigh_joint",
    "lf_calf_joint", "lh_calf_joint", "rf_calf_joint", "rh_calf_joint",
]
Q0_VALUES = np.array([
    0.06,  0.06, -0.06, -0.06,
    0.85,  0.85,  0.85,  0.85,
   -1.75, -1.75, -1.75, -1.75,
], dtype=np.float32)
ROBOT_SPAWN = (0.0, 0.0, 0.12)

CAM_FORWARD_OFFSET = 0.10
CAM_UP_OFFSET = 0.05
CAM_LOOKAT_DIST = 1.0
SIM_DECIMATION = 4
KP, KV = 5.0, 0.5
ACTION_SCALE = 0.30
COLLISION_MARGIN = 0.22
MIN_Z = 0.05
BEACON_CAPTURE_DIST = 0.35

WORLD_MIN = np.array([-3.5, -3.5], dtype=np.float32)
WORLD_MAX = np.array([3.5,   3.5], dtype=np.float32)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def parse_args():
    p = argparse.ArgumentParser(description="Evaluate CEM planner in Genesis")
    p.add_argument("--ppo_ckpt", type=str, required=True)
    p.add_argument("--lewm_ckpt", type=str, required=True)
    p.add_argument("--energy_ckpt", type=str, required=True)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--sim_backend", type=str, default="auto")
    p.add_argument("--max_steps", type=int, default=2000)
    p.add_argument("--n_episodes", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no_video", action="store_true")
    p.add_argument("--out", type=str, default="eval_results/cem_eval.mp4")
    # CEM / exploration config
    p.add_argument("--horizon", type=int, default=12)
    p.add_argument("--n_candidates", type=int, default=128)
    p.add_argument("--n_elites", type=int, default=16)
    p.add_argument("--n_iterations", type=int, default=3)
    p.add_argument("--energy_weight", type=float, default=1.0)
    p.add_argument("--stall_penalty", type=float, default=0.5)
    p.add_argument("--coverage_weight", type=float, default=0.3)
    p.add_argument("--collision_fwd_penalty", type=float, default=8.0)
    p.add_argument("--collision_yaw_bonus", type=float, default=4.0)
    p.add_argument("--coverage_dim", type=int, default=3)
    p.add_argument("--coverage_cells", type=int, default=8)
    # Model config
    p.add_argument("--latent_dim", type=int, default=192)
    p.add_argument("--image_size", type=int, default=224)
    p.add_argument("--patch_size", type=int, default=14)
    p.add_argument("--use_proprio", action="store_true")
    # Scene
    p.add_argument("--n_beacons", type=int, default=3)
    p.add_argument("--n_distractors", type=int, default=1)
    return p.parse_args()


# --------------------------------------------------------------------------- #
# Model loading
# --------------------------------------------------------------------------- #

def load_world_model(ckpt_path, args, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    sd = clean_state_dict(ckpt["model_state_dict"])
    model = LeWorldModel(
        latent_dim=args.latent_dim,
        image_size=args.image_size,
        patch_size=args.patch_size,
        use_proprio=args.use_proprio,
    )
    model.load_state_dict(sd, strict=False)
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def load_energy_head(ckpt_path, args, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    sd = {k.replace("_orig_mod.", ""): v for k, v in ckpt["head_state_dict"].items()}
    head = LatentEnergyHead(latent_dim=args.latent_dim)
    head.load_state_dict(sd)
    head = head.to(device).eval()
    for p in head.parameters():
        p.requires_grad_(False)
    return head


def load_ppo(ckpt_path, device):
    model = ActorCritic(obs_dim=50, act_dim=12).to(device)
    ppo_sd = load_ppo_checkpoint(ckpt_path, device=device)
    model.load_state_dict(ppo_sd, strict=False)
    model.eval()
    return model


# --------------------------------------------------------------------------- #
# Camera helpers
# --------------------------------------------------------------------------- #

def move_cams(robot, cam_brain, cam_eye, cam_over):
    """Position all 3 cameras. Returns (robot_pos_3d, robot_yaw)."""
    p = to_numpy(robot.get_pos())
    q = to_numpy(robot.get_quat())
    if p.ndim > 1: p = p[0]
    if q.ndim > 1: q = q[0]
    fw, up = forward_up_from_quat(q)

    # First-person cameras
    brain_pos = p + fw * CAM_FORWARD_OFFSET + up * CAM_UP_OFFSET
    brain_lk = brain_pos + fw * CAM_LOOKAT_DIST
    cam_brain.set_pose(pos=brain_pos, lookat=brain_lk, up=up)
    cam_eye.set_pose(pos=brain_pos, lookat=brain_lk, up=up)

    # Third-person chase-cam
    over_pos = p - fw * 1.8 + np.array([0.0, 0.0, 1.0], dtype=np.float32)
    over_lk = p + fw * 0.45
    cam_over.set_pose(
        pos=over_pos, lookat=over_lk,
        up=np.array([0.0, 0.0, 1.0], dtype=np.float32),
    )

    yaw = math.atan2(float(fw[1]), float(fw[0]))
    return p, yaw


def render_rgb(cam) -> np.ndarray:
    """Return HxWx3 uint8 RGB from any Genesis camera."""
    out = cam.render()
    arr = None
    if isinstance(out, (tuple, list)):
        for item in out:
            a = np.asarray(to_numpy(item))
            if a.ndim == 3 and a.shape[-1] >= 3:
                arr = a[..., :3]; break
    elif isinstance(out, dict):
        for k in ("rgb", "color", "image"):
            if k in out:
                arr = np.asarray(out[k])[..., :3]; break
    else:
        arr = np.asarray(to_numpy(out))[..., :3]
    if arr is None:
        raise RuntimeError("Camera render failed")
    if arr.dtype != np.uint8:
        mx = float(np.nanmax(arr)) if arr.size else 1.0
        arr = np.clip(arr * (255.0 / mx if mx > 1.0 else 255.0), 0, 255).astype(np.uint8)
    return arr


# --------------------------------------------------------------------------- #
# Scene helpers
# --------------------------------------------------------------------------- #

def build_eval_scene(gs, seed, n_beacons, n_distractors):
    rng = np.random.RandomState(seed)
    maze_style = rng.choice(MAZE_STYLES)
    obstacle_layout, beacon_layout = generate_composite_scene(
        seed=seed, maze_style=maze_style,
        n_free_obstacles=2, n_beacons=n_beacons, n_distractors=n_distractors,
    )
    return obstacle_layout, beacon_layout, maze_style


def sample_safe_spawn(layout, rng, clearance=0.40, spawn_range=2.0, max_attempts=500):
    for _ in range(max_attempts):
        xy = rng.uniform(-spawn_range, spawn_range, size=(1, 2))
        if not detect_collisions(torch.from_numpy(xy).float(), layout, margin=clearance)[0]:
            yaw = rng.uniform(0, 2 * math.pi)
            return float(xy[0, 0]), float(xy[0, 1]), yaw
    return 0.0, 0.0, 0.0


def beacon_distances(robot_xy, beacon_layout):
    dists = []
    for b in beacon_layout.beacons:
        dx = robot_xy[0] - b.pos[0]
        dy = robot_xy[1] - b.pos[1]
        dists.append(math.sqrt(dx * dx + dy * dy))
    return np.array(dists, dtype=np.float32)


def beacon_color_rgb(beacon):
    """Get (R, G, B) int tuple for a beacon's identity."""
    base = BEACON_FAMILIES.get(beacon.identity, (0.5, 0.5, 0.5))
    return tuple(int(c * 255) for c in base)


# --------------------------------------------------------------------------- #
# HUD composition (matches TinyQuadJEPA-v2 layout: 896×660)
# --------------------------------------------------------------------------- #

def _wp_to_map(xy, mx, my, mw, mh):
    nx = (float(xy[0]) - float(WORLD_MIN[0])) / max(float(WORLD_MAX[0] - WORLD_MIN[0]), 1e-8)
    ny = (float(xy[1]) - float(WORLD_MIN[1])) / max(float(WORLD_MAX[1] - WORLD_MIN[1]), 1e-8)
    return (mx + int(np.clip(nx, 0, 1) * mw), my + mh - int(np.clip(ny, 0, 1) * mh))


class SpatialCoverageGrid:
    """Ground-truth spatial coverage grid (diagnostic only, robot does NOT see this)."""

    def __init__(self, world_min, world_max, cell_size=0.25):
        self.world_min = np.array(world_min, dtype=np.float32)
        self.world_max = np.array(world_max, dtype=np.float32)
        self.cell_size = cell_size
        extent = self.world_max - self.world_min
        self.nx = max(1, int(extent[0] / cell_size))
        self.ny = max(1, int(extent[1] / cell_size))
        self.counts = np.zeros((self.nx, self.ny), dtype=np.int32)

    def mark(self, x: float, y: float):
        ix = int((x - self.world_min[0]) / self.cell_size)
        iy = int((y - self.world_min[1]) / self.cell_size)
        ix = max(0, min(ix, self.nx - 1))
        iy = max(0, min(iy, self.ny - 1))
        self.counts[ix, iy] += 1

    @property
    def cells_visited(self) -> int:
        return int((self.counts > 0).sum())

    @property
    def coverage_frac(self) -> float:
        return self.cells_visited / max(1, self.nx * self.ny)

    def to_heatmap(self, width: int, height: int) -> np.ndarray:
        """Render as (H, W, 3) uint8 heatmap image."""
        log_counts = np.log1p(self.counts.T[::-1].astype(np.float32))
        mx = log_counts.max()
        if mx > 0:
            normed = log_counts / mx
        else:
            normed = log_counts
        # Green channel for visited areas
        r = (normed * 40).astype(np.uint8)
        g = (normed * 200).astype(np.uint8)
        b = (normed * 60).astype(np.uint8)
        img = np.stack([r, g, b], axis=-1)
        return np.array(Image.fromarray(img).resize((width, height), Image.NEAREST))


def draw_minimap(draw, canvas, robot_xy, robot_yaw, trail, plan_path,
                 beacon_layout, captured, spatial_cov=None,
                 mx=514, my=494, mw=372, mh=155):
    """Draw minimap with coverage heatmap, trail, beacon markers, and robot."""
    draw.rectangle([mx, my, mx + mw, my + mh], fill=(18, 18, 18), outline=(95, 95, 95))

    # Coverage heatmap overlay (diagnostic — from ground-truth positions)
    if spatial_cov is not None and spatial_cov.cells_visited > 0:
        heatmap = spatial_cov.to_heatmap(mw, mh)
        hm_img = Image.fromarray(heatmap)
        # Blend onto canvas at minimap position
        bg = canvas.crop((mx, my, mx + mw, my + mh))
        blended = Image.blend(bg, hm_img, alpha=0.6)
        canvas.paste(blended, (mx, my))

    # Trail (yellow)
    if len(trail) > 1:
        pts = [_wp_to_map(t, mx, my, mw, mh) for t in trail[-300:]]
        draw.line(pts, fill=(255, 220, 80), width=2)

    # Planned path (blue)
    if plan_path is not None and len(plan_path) > 1:
        pts = [_wp_to_map(t, mx, my, mw, mh) for t in plan_path]
        draw.line(pts, fill=(0, 170, 255), width=3)

    # Beacon markers
    for i, b in enumerate(beacon_layout.beacons):
        bxy = np.array([b.pos[0], b.pos[1]])
        px, py = _wp_to_map(bxy, mx, my, mw, mh)
        col = beacon_color_rgb(b)
        if captured[i]:
            draw.ellipse([px - 6, py - 6, px + 6, py + 6],
                         fill=col, outline=(255, 255, 255), width=2)
        else:
            draw.ellipse([px - 5, py - 5, px + 5, py + 5],
                         fill=(60, 60, 60), outline=col, width=2)

    # Robot (white triangle)
    rx, ry = _wp_to_map(robot_xy, mx, my, mw, mh)
    fw = np.array([math.cos(robot_yaw), math.sin(robot_yaw)])
    lf = np.array([math.cos(robot_yaw + 2.5), math.sin(robot_yaw + 2.5)])
    rt = np.array([math.cos(robot_yaw - 2.5), math.sin(robot_yaw - 2.5)])
    s = 10.0
    tri = [
        (rx + int(fw[0] * s), ry - int(fw[1] * s)),
        (rx + int(lf[0] * s * 0.8), ry - int(lf[1] * s * 0.8)),
        (rx + int(rt[0] * s * 0.8), ry - int(rt[1] * s * 0.8)),
    ]
    draw.polygon(tri, fill=(255, 255, 255), outline=(10, 10, 10))


def compose_frame(
    over_rgb: np.ndarray,
    eye_rgb: np.ndarray,
    robot_xy: np.ndarray,
    robot_yaw: float,
    trail: list,
    plan_path,
    beacon_layout: BeaconLayout,
    captured: list,
    status_lines: list,
    energy_val: float,
    energy_history: list,
    event_log: list,
    spatial_cov=None,
) -> np.ndarray:
    """Compose 896×660 HUD frame matching TinyQuadJEPA-v2 layout."""
    # Layout:
    #   Header  y=  0..55   — title + beacon completion pills
    #   Views   y= 56..567  — overhead (0..511) | eye (512..895, 56..439)
    #   R-panel y=440..659  — status text + minimap
    #   L-strip y=568..659  — event log
    canvas = Image.new("RGB", (896, 660), (20, 20, 20))
    draw = ImageDraw.Draw(canvas)

    # ── Header ──────────────────────────────────────────────── #
    draw.rectangle([0, 0, 895, 55], fill=(12, 12, 12), outline=(55, 55, 55))
    draw.text((12, 19), "LeWM  CEM Planner Eval", fill=(0, 220, 140))

    # Beacon completion pills
    n_beacons = len(beacon_layout.beacons)
    pill_x0, pill_w, pill_gap = 280, 94, 5
    for i, b in enumerate(beacon_layout.beacons):
        bx = pill_x0 + i * (pill_w + pill_gap)
        col = beacon_color_rgb(b)
        label = b.identity
        if captured[i]:
            draw.rectangle([bx, 10, bx + pill_w, 45], fill=col, outline=(255, 255, 255), width=2)
            draw.text((bx + 6, 21), f"[OK] {label}", fill=(0, 0, 0))
        else:
            draw.rectangle([bx, 10, bx + pill_w, 45], fill=(25, 25, 25), outline=(55, 55, 55), width=1)
            draw.text((bx + 6, 21), label, fill=(65, 65, 65))

    n_found = sum(captured)
    draw.text((pill_x0 + n_beacons * (pill_w + pill_gap) + 8, 21),
              f"{n_found}/{n_beacons}", fill=(160, 160, 160))

    # ── Overhead view (left) ────────────────────────────────── #
    canvas.paste(Image.fromarray(over_rgb[:, :, :3].astype(np.uint8)), (0, 56))

    # ── Eye view (right top) ────────────────────────────────── #
    canvas.paste(
        Image.fromarray(eye_rgb[:, :, :3].astype(np.uint8)).resize((384, 384)),
        (512, 56),
    )

    # ── Right HUD panel (below eye view) ────────────────────── #
    draw.rectangle([512, 440, 895, 659], fill=(14, 14, 14), outline=(55, 55, 55))
    for i, line in enumerate(status_lines):
        draw.text((520, 447 + i * 16), line, fill=(200, 200, 200))

    # Energy bar
    bar_y = 447 + len(status_lines) * 16 + 4
    bar_w = 360
    draw.rectangle([520, bar_y, 520 + bar_w, bar_y + 12], fill=(30, 30, 30), outline=(55, 55, 55))
    frac = min(1.0, energy_val / 0.8)
    r = int(255 * frac)
    g = int(255 * (1.0 - frac))
    draw.rectangle([520, bar_y, 520 + int(frac * bar_w), bar_y + 12], fill=(r, g, 40))
    draw.text((520, bar_y + 14), f"energy: {energy_val:.3f}", fill=(160, 160, 160))

    # Energy history trace
    if len(energy_history) > 2:
        trace_y0, trace_h = bar_y + 34, 30
        recent = energy_history[-80:]
        e_min = min(recent)
        e_max = max(max(recent), e_min + 0.01)
        pts = []
        for j, e in enumerate(recent):
            px = 520 + int(j / max(len(recent) - 1, 1) * bar_w)
            py = trace_y0 + trace_h - int((e - e_min) / (e_max - e_min) * trace_h)
            pts.append((px, py))
        if len(pts) > 1:
            draw.line(pts, fill=(0, 200, 255), width=1)

    # Minimap with coverage overlay
    draw_minimap(draw, canvas, robot_xy, robot_yaw, trail, plan_path,
                 beacon_layout, captured, spatial_cov=spatial_cov)

    # ── Event log (below overhead view) ─────────────────────── #
    draw.rectangle([0, 568, 511, 659], fill=(14, 14, 14), outline=(55, 55, 55))
    draw.text((8, 572), "events", fill=(70, 70, 70))
    for i, (ev_type, ev_text) in enumerate((event_log or [])[-4:]):
        if ev_type == "CAPTURE":
            col = (255, 210, 60)
        elif ev_type == "STUCK":
            col = (255, 100, 100)
        else:
            col = (80, 255, 120)
        draw.text((8, 587 + i * 18), ev_text, fill=col)

    return np.asarray(canvas)


# --------------------------------------------------------------------------- #
# PPO step helper
# --------------------------------------------------------------------------- #

def get_ppo_obs(robot, q0_t, prev_action, cmd, act_dofs, device):
    """Build PPO observation tensor."""
    pos = robot.get_pos().to(device)
    quat = robot.get_quat().to(device)
    vel = robot.get_vel().to(device)
    ang = robot.get_ang().to(device)
    pos, quat, vel, ang = [x.unsqueeze(0) if x.dim() == 1 else x
                           for x in (pos, quat, vel, ang)]
    q = robot.get_dofs_position(act_dofs).to(device)
    dq = robot.get_dofs_velocity(act_dofs).to(device)
    q = q.unsqueeze(0) if q.dim() == 1 else q
    dq = dq.unsqueeze(0) if dq.dim() == 1 else dq

    # Add noise
    proprio = torch.cat([pos[:, 2:3], quat,
                         world_to_body_vec(quat, vel),
                         world_to_body_vec(quat, ang),
                         q - q0_t.unsqueeze(0), dq, prev_action], dim=1)
    noise = torch.randn_like(proprio) * 0.01
    noise[:, 1:5] *= 2.0
    noise[:, 5:11] *= 5.0
    proprio_noisy = proprio + noise

    return torch.cat([proprio_noisy, cmd], dim=1), proprio_noisy


# --------------------------------------------------------------------------- #
# Main eval
# --------------------------------------------------------------------------- #

@torch.no_grad()
def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    # Load models
    print("Loading models...")
    world_model = load_world_model(args.lewm_ckpt, args, device)
    energy_head = load_energy_head(args.energy_ckpt, args, device)
    print(f"  World model: {args.lewm_ckpt}")
    print(f"  Energy head: {args.energy_ckpt}")

    cem_config = CEMConfig(
        horizon=args.horizon,
        n_candidates=args.n_candidates,
        n_elites=args.n_elites,
        n_iterations=args.n_iterations,
        energy_weight=args.energy_weight,
        stall_penalty=args.stall_penalty,
        coverage_weight=args.coverage_weight,
        collision_fwd_penalty=args.collision_fwd_penalty,
        collision_yaw_bonus=args.collision_yaw_bonus,
        coverage_dim=args.coverage_dim,
        coverage_cells=args.coverage_cells,
    )
    planner = CEMPlanner(world_model, energy_head, config=cem_config, device=device)

    # Init simulator
    import genesis as gs
    init_genesis_once(args.sim_backend)
    ppo_model = load_ppo(args.ppo_ckpt, gs.device)
    print(f"  PPO policy:  {args.ppo_ckpt}")

    all_results = []

    for ep in range(args.n_episodes):
        ep_seed = args.seed + ep
        rng = np.random.RandomState(ep_seed)
        print(f"\n{'=' * 60}")
        print(f"Episode {ep + 1}/{args.n_episodes} (seed={ep_seed})")
        print(f"{'=' * 60}")

        # Build scene
        obstacle_layout, beacon_layout, maze_style = build_eval_scene(
            gs, ep_seed, args.n_beacons, args.n_distractors,
        )
        print(f"  Scene: {maze_style} | {len(obstacle_layout.obstacles)} obstacles "
              f"| {len(beacon_layout.beacons)} beacons")

        scene = gs.Scene(show_viewer=False)
        scene.add_entity(gs.morphs.Plane())
        robot = scene.add_entity(
            gs.morphs.URDF(file=URDF_PATH, pos=ROBOT_SPAWN, fixed=False),
        )
        add_obstacles_to_scene(scene, obstacle_layout)
        for obs_spec in beacon_layout.all_obstacles():
            scene.add_entity(
                gs.morphs.Box(pos=obs_spec.pos, size=obs_spec.size, fixed=True),
                surface=gs.surfaces.Rough(color=obs_spec.color),
            )

        # Three cameras matching TinyQuadJEPA-v2
        cam_brain = scene.add_camera(res=(args.image_size, args.image_size), fov=58, GUI=False)
        cam_eye = scene.add_camera(res=(384, 384), fov=58, GUI=False)
        cam_over = scene.add_camera(res=(512, 512), fov=55, GUI=False)

        scene.build(n_envs=1)

        name_to_joint = {j.name: j for j in robot.joints}
        dof_idx = [list(name_to_joint[jn].dofs_idx_local)[0] for jn in JOINTS_ACTUATED]
        act_dofs = torch.tensor(dof_idx, device=gs.device, dtype=torch.int64)
        q0_np = Q0_VALUES
        q0_t = torch.from_numpy(q0_np).to(gs.device)

        robot.set_dofs_kp(torch.ones(12, device=gs.device) * KP, act_dofs)
        robot.set_dofs_kv(torch.ones(12, device=gs.device) * KV, act_dofs)

        # Spawn at safe location
        sx, sy, syaw = sample_safe_spawn(obstacle_layout, rng)
        spawn_pos = torch.tensor([[sx, sy, ROBOT_SPAWN[2]]], device=gs.device)
        spawn_quat = torch.tensor(
            [[math.cos(syaw / 2), 0, 0, math.sin(syaw / 2)]],
            device=gs.device,
        )
        robot.set_pos(spawn_pos, zero_velocity=True)
        robot.set_quat(spawn_quat, zero_velocity=True)
        robot.set_dofs_position(q0_t.unsqueeze(0), act_dofs)
        robot.set_dofs_velocity(torch.zeros((1, 12), device=gs.device), act_dofs)

        for _ in range(20):
            scene.step()

        # Episode state
        planner.reset()
        prev_action = torch.zeros((1, 12), device=gs.device)
        latency_buf = torch.zeros((2, 1, 3), device=gs.device)

        n_beacons = len(beacon_layout.beacons)
        captured = [False] * n_beacons
        total_collisions = 0
        trail: List[np.ndarray] = []
        energy_history: List[float] = []
        event_log: List[Tuple[str, str]] = []

        # Ground-truth spatial coverage (diagnostic only — robot doesn't see it)
        spatial_cov = SpatialCoverageGrid(WORLD_MIN, WORLD_MAX, cell_size=0.25)

        writer = None
        if not args.no_video:
            vid_path = args.out.replace(".mp4", f"_ep{ep:03d}.mp4")
            writer = imageio.get_writer(vid_path, fps=30)

        t0 = time.time()

        for step in range(args.max_steps):
            # Position cameras
            robot_pos_3d, robot_yaw = move_cams(robot, cam_brain, cam_eye, cam_over)
            robot_xy = robot_pos_3d[:2].astype(np.float32)
            trail.append(robot_xy.copy())
            spatial_cov.mark(float(robot_xy[0]), float(robot_xy[1]))

            # Render brain camera for JEPA encoder
            brain_rgb = render_rgb(cam_brain)
            vis_t = (
                torch.from_numpy(np.transpose(brain_rgb[:, :, :3], (2, 0, 1)).copy())
                .float().to(device).unsqueeze(0) / 255.0
            )

            # Proprio for planner
            ppo_obs, proprio_noisy = get_ppo_obs(
                robot, q0_t, prev_action,
                torch.zeros((1, 3), device=gs.device), act_dofs, gs.device,
            )
            prop_for_planner = proprio_noisy[:, :47] if args.use_proprio else None

            # CEM plan
            cmd = planner.step(vis_t, prop_for_planner)
            cmd_3d = cmd.unsqueeze(0).to(gs.device)

            # Log stuck recovery events
            if planner.is_stuck:
                event_log.append(
                    ("STUCK", f"step {step:4d}  STUCK — random yaw escape "
                              f"(event #{planner.stuck_events})")
                )
                if step % 100 != 0:  # avoid double-print on progress lines
                    print(f"  [STUCK] step {step} — escape #{planner.stuck_events}")

            # Current energy (for HUD)
            with torch.no_grad():
                z_proj = world_model.encode_observation(vis_t, prop_for_planner)
                cur_energy = float(energy_head(z_proj).item())
            energy_history.append(cur_energy)

            # Latency buffer
            latency_buf = torch.roll(latency_buf, shifts=-1, dims=0)
            latency_buf[-1] = cmd_3d
            active_cmd = latency_buf[0]

            # PPO step
            ppo_obs_full, _ = get_ppo_obs(
                robot, q0_t, prev_action, active_cmd, act_dofs, gs.device,
            )
            actions = ppo_model.act_deterministic(ppo_obs_full)
            prev_action = actions.clone()

            q_tgt = q0_t.unsqueeze(0) + ACTION_SCALE * actions
            q_tgt[:, 0:4] = torch.clamp(q_tgt[:, 0:4], -0.8, 0.8)
            q_tgt[:, 4:8] = torch.clamp(q_tgt[:, 4:8], -1.5, 1.5)
            q_tgt[:, 8:12] = torch.clamp(q_tgt[:, 8:12], -2.5, -0.5)
            robot.control_dofs_position(q_tgt, act_dofs)
            for _ in range(SIM_DECIMATION):
                scene.step()

            # Collision check — also feed to planner as proprioceptive signal
            pos_after = robot.get_pos()
            colliding = detect_collisions(
                pos_after[:, :2], obstacle_layout, margin=COLLISION_MARGIN,
            )
            is_colliding = bool(colliding[0])
            planner.report_collision(is_colliding)
            if is_colliding:
                total_collisions += 1

            # Beacon capture
            if n_beacons > 0:
                dists = beacon_distances(robot_xy, beacon_layout)
                for bi in range(n_beacons):
                    if not captured[bi] and dists[bi] < BEACON_CAPTURE_DIST:
                        captured[bi] = True
                        b = beacon_layout.beacons[bi]
                        n_found = sum(captured)
                        print(f"  [CAPTURE] {b.identity} beacon at step {step} "
                              f"dist={dists[bi]:.2f}m ({n_found}/{n_beacons})")
                        event_log.append(
                            ("CAPTURE", f"step {step:4d}  CAPTURED {b.identity}  "
                                        f"({n_found}/{n_beacons})")
                        )

            # Fall check
            if float(pos_after[0, 2]) < MIN_Z:
                event_log.append(("FALL", f"step {step:4d}  Robot fell"))
                break

            # Compose HUD frame (every 2nd step to reduce camera overhead)
            if writer is not None and step % 2 == 0:
                over_rgb = render_rgb(cam_over)
                eye_rgb = render_rgb(cam_eye)

                cov = planner.coverage
                cov_cells = cov.cells_visited if cov else 0
                stuck_tag = " [STUCK]" if planner.is_stuck else (" [TURNING]" if planner.is_turning else "")
                status_lines = [
                    f"step: {step}/{args.max_steps}{stuck_tag}",
                    f"pos: ({robot_xy[0]:.2f}, {robot_xy[1]:.2f})  yaw: {math.degrees(robot_yaw):.0f}deg",
                    f"cmd: ({float(cmd[0]):.2f}, {float(cmd[1]):.2f}, {float(cmd[2]):.2f})",
                    f"collisions: {total_collisions}  escapes: {planner.stuck_events}",
                    f"beacons: {sum(captured)}/{n_beacons}",
                    f"latent cells: {cov_cells}",
                ]

                frame = compose_frame(
                    over_rgb, eye_rgb,
                    robot_xy, robot_yaw, trail, None,
                    beacon_layout, captured,
                    status_lines, cur_energy, energy_history,
                    event_log, spatial_cov=spatial_cov,
                )
                writer.append_data(frame)

            # Progress
            if step % 100 == 0:
                cov = planner.coverage
                lcells = cov.cells_visited if cov else 0
                scov = spatial_cov.coverage_frac * 100
                stuck_tag = " STUCK" if planner.is_stuck else (" TURNING" if planner.is_turning else "")
                d = planner.diag
                print(f"  step {step:4d} | energy={cur_energy:.3f} | "
                      f"beacons={sum(captured)}/{n_beacons} | col={total_collisions} | "
                      f"lcells={lcells} | coverage={scov:.1f}% | "
                      f"escapes={planner.stuck_events}{stuck_tag}")
                if d:
                    print(f"    CEM: cost={d.get('cost_mean',0):.2f}±{d.get('cost_std',0):.3f} "
                          f"energy={d.get('energy_mean',0):.2f}±{d.get('energy_std',0):.3f} "
                          f"z_pred_std={d.get('z_pred_std',0):.4f} "
                          f"elite_vx={d.get('elite_vx',0):.2f} yaw={d.get('elite_yaw',0):.2f}")

            # All beacons found
            if all(captured):
                print(f"\n  All {n_beacons} beacons captured at step {step}!")
                break

        elapsed = time.time() - t0
        n_steps = step + 1

        if writer is not None:
            writer.close()
            print(f"  Video saved: {vid_path}")

        trajectory = np.array(trail)
        path_length = float(np.sum(np.linalg.norm(np.diff(trajectory, axis=0), axis=1))) if len(trail) > 1 else 0.0

        cov = planner.coverage
        lcells = cov.cells_visited if cov else 0
        scov_pct = spatial_cov.coverage_frac * 100
        scov_cells = spatial_cov.cells_visited
        print(f"\n  Results:")
        print(f"    Steps:          {n_steps}/{args.max_steps}")
        print(f"    Beacons:        {sum(captured)}/{n_beacons}")
        print(f"    Collisions:     {total_collisions}")
        print(f"    Path len:       {path_length:.2f} m")
        print(f"    Latent cells:   {lcells}")
        print(f"    Spatial cells:  {scov_cells}/{spatial_cov.nx * spatial_cov.ny} ({scov_pct:.1f}%)")
        print(f"    Stuck escapes:  {planner.stuck_events}")
        print(f"    Time:           {elapsed:.1f}s ({n_steps / elapsed:.1f} steps/s)")

        traj_path = os.path.join(os.path.dirname(args.out), f"trajectory_ep{ep:03d}.npy")
        np.save(traj_path, trajectory)

        all_results.append({
            "episode": ep, "seed": ep_seed, "maze_style": maze_style,
            "steps": n_steps, "beacons_captured": sum(captured),
            "beacons_total": n_beacons, "collisions": total_collisions,
            "path_length": path_length, "latent_cells": lcells,
            "spatial_coverage_pct": round(scov_pct, 1),
            "spatial_cells": scov_cells,
            "elapsed": elapsed,
        })

        scene.destroy()

    # Summary
    print(f"\n{'=' * 60}")
    print("Summary")
    print(f"{'=' * 60}")
    total_cap = sum(r["beacons_captured"] for r in all_results)
    total_pos = sum(r["beacons_total"] for r in all_results)
    total_col = sum(r["collisions"] for r in all_results)
    avg_path = np.mean([r["path_length"] for r in all_results])
    avg_steps = np.mean([r["steps"] for r in all_results])

    avg_scov = np.mean([r["spatial_coverage_pct"] for r in all_results])
    print(f"  Episodes:     {args.n_episodes}")
    print(f"  Beacons:      {total_cap}/{total_pos} ({100 * total_cap / max(1, total_pos):.0f}%)")
    print(f"  Collisions:   {total_col} total ({total_col / args.n_episodes:.1f}/ep)")
    print(f"  Avg path:     {avg_path:.2f} m")
    print(f"  Avg coverage: {avg_scov:.1f}%")
    print(f"  Avg steps:    {avg_steps:.0f}")

    summary_path = os.path.join(os.path.dirname(args.out), "eval_summary.json")
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Results saved to {summary_path}")


if __name__ == "__main__":
    main()
