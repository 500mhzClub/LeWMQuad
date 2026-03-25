#!/usr/bin/env python3
"""Generate diagnostic visuals for leWMQuad data pipeline outputs.

Produces a set of PNG figures in --out_dir covering:
  1. Egocentric RGB frame grids (from .h5)
  2. Top-down trajectory plots (from .npz base_pos)
  3. Label distribution histograms (clearance, traversability, beacon stats)
  4. Per-chunk summary heatmap

Usage:
    python scripts/visualize_data.py --raw_dir jepa_raw_data --h5_dir jepa_final_dataset
    python scripts/visualize_data.py --h5_dir jepa_final_dataset --out_dir figs/
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import h5py
import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #

COMMAND_PATTERN_NAMES = [
    "OU explore", "retreat", "stop", "recovery",
    "dead-end backout", "wall-follow", "spin-in-place", "unknown",
]


def savefig(fig, path, dpi=120):
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {path}")


def load_h5_sample(path, n_env_samples=16, t_samples=8):
    """Return sampled vision frames and label arrays from one h5 file."""
    with h5py.File(path, "r") as f:
        n_envs = f["vision"].shape[0]
        T      = f["vision"].shape[1]

        env_idx = np.linspace(0, n_envs - 1, n_env_samples, dtype=int)
        t_idx   = np.linspace(0, T - 1, t_samples, dtype=int)

        # vision: (n_env_samples, t_samples, 3, H, W)
        vision = np.stack([f["vision"][e, t_idx] for e in env_idx])

        labels = {}
        for field in ("clearance", "traversability", "beacon_visible",
                      "beacon_identity", "beacon_bearing", "beacon_range",
                      "cmd_pattern", "dones", "collisions"):
            if field in f:
                labels[field] = f[field][:]  # full array — these are small

    return vision, labels, n_envs, T


def load_npz_sample(path, n_env_samples=64):
    data = np.load(path, allow_pickle=True)
    n_envs = data["base_pos"].shape[0]
    idx = np.linspace(0, n_envs - 1, min(n_env_samples, n_envs), dtype=int)
    out = {
        "base_pos":    data["base_pos"][idx],
        "clearance":   data["clearance"][idx],
        "traversability": data["traversability"][idx],
        "beacon_visible": data["beacon_visible"][idx],
        "beacon_range":   data["beacon_range"][idx],
        "cmd_pattern":    data["cmd_pattern"][idx],
        "collisions":     data["collisions"][idx],
        "dones":          data["dones"][idx],
    }
    data.close()
    return out, n_envs


# ------------------------------------------------------------------ #
# Figure 1 — Egocentric RGB grid
# ------------------------------------------------------------------ #

def fig_rgb_grid(h5_files, out_dir, n_chunks=4, n_envs=8, n_times=6):
    """Grid: chunks (rows of rows) × envs (rows) × timesteps (cols)."""
    files = h5_files[:n_chunks]
    if not files:
        return

    fig_rows = len(files) * n_envs
    fig_cols = n_times
    fig, axes = plt.subplots(fig_rows, fig_cols,
                             figsize=(fig_cols * 1.6, fig_rows * 1.6))
    axes = np.array(axes).reshape(fig_rows, fig_cols)

    for ci, path in enumerate(files):
        chunk_name = os.path.basename(path).replace("_rgb.h5", "")
        vision, _, n_envs_total, T = load_h5_sample(path, n_envs, n_times)
        # vision: (n_envs, n_times, 3, H, W)

        for ei in range(n_envs):
            row = ci * n_envs + ei
            for ti in range(n_times):
                ax = axes[row, ti]
                img = vision[ei, ti]           # (3, H, W)
                img = np.transpose(img, (1, 2, 0))  # (H, W, 3)
                ax.imshow(img)
                ax.axis("off")
                if ti == 0:
                    ax.set_ylabel(f"{chunk_name}\nenv {ei}", fontsize=5, rotation=0,
                                  labelpad=40, va="center")

        # timestep labels on top row of each chunk
        t_idx = np.linspace(0, T - 1, n_times, dtype=int)
        if ci == 0:
            for ti, t in enumerate(t_idx):
                axes[0, ti].set_title(f"t={t}", fontsize=7)

    fig.suptitle("Egocentric RGB samples — first 4 chunks", fontsize=10, y=1.01)
    plt.tight_layout(pad=0.3)
    savefig(fig, os.path.join(out_dir, "01_rgb_grid.png"))


# ------------------------------------------------------------------ #
# Figure 2 — Top-down trajectory plots
# ------------------------------------------------------------------ #

def fig_trajectories(npz_files, out_dir, n_chunks=6, n_envs=48):
    files = npz_files[:n_chunks]
    if not files:
        return

    ncols = 3
    nrows = int(np.ceil(len(files) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 4))
    axes = np.array(axes).reshape(-1)

    for ci, path in enumerate(files):
        ax = axes[ci]
        chunk_name = os.path.basename(path).replace(".npz", "")
        sample, n_envs_total = load_npz_sample(path, n_envs)

        pos = sample["base_pos"]   # (n_envs, steps, 3)
        cl  = sample["clearance"]  # (n_envs, steps)

        # Colour each trajectory by mean clearance
        cmap = plt.cm.RdYlGn
        mean_cl = np.clip(cl.mean(axis=1), 0, 2.0)
        norm = Normalize(vmin=0, vmax=2.0)

        for ei in range(pos.shape[0]):
            x, y = pos[ei, :, 0], pos[ei, :, 1]
            color = cmap(norm(mean_cl[ei]))
            ax.plot(x, y, lw=0.4, alpha=0.6, color=color)

        ax.set_title(f"{chunk_name}  (n={n_envs_total})", fontsize=8)
        ax.set_xlabel("x (m)", fontsize=7)
        ax.set_ylabel("y (m)", fontsize=7)
        ax.tick_params(labelsize=6)
        ax.set_aspect("equal")

    # Hide unused axes
    for ax in axes[len(files):]:
        ax.axis("off")

    # Shared colorbar
    sm = ScalarMappable(cmap=plt.cm.RdYlGn, norm=Normalize(0, 2.0))
    sm.set_array([])
    fig.colorbar(sm, ax=axes[:len(files)].tolist(), shrink=0.6,
                 label="Mean clearance (m)")
    fig.suptitle("Top-down trajectories (coloured by clearance)", fontsize=11)
    plt.tight_layout()
    savefig(fig, os.path.join(out_dir, "02_trajectories.png"))


# ------------------------------------------------------------------ #
# Figure 3 — Label distributions
# ------------------------------------------------------------------ #

def fig_label_distributions(npz_files, out_dir):
    if not npz_files:
        return

    # Aggregate across all chunks (sample envs per chunk for speed)
    agg = {k: [] for k in ("clearance", "traversability", "beacon_range",
                            "beacon_visible", "cmd_pattern", "collisions")}

    for path in npz_files:
        sample, _ = load_npz_sample(path, n_env_samples=32)
        agg["clearance"].append(sample["clearance"].ravel())
        agg["traversability"].append(sample["traversability"].ravel())
        agg["beacon_range"].append(
            sample["beacon_range"][sample["beacon_visible"].astype(bool)].ravel()
        )
        agg["beacon_visible"].append(sample["beacon_visible"].ravel().astype(float))
        agg["cmd_pattern"].append(sample["cmd_pattern"].ravel())
        agg["collisions"].append(sample["collisions"].ravel().astype(float))

    for k in agg:
        agg[k] = np.concatenate(agg[k])

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))

    # Clearance
    ax = axes[0, 0]
    cl = agg["clearance"]
    cl_clipped = np.clip(cl[cl < 900], 0, 4.0)
    ax.hist(cl_clipped, bins=60, color="#4C72B0", edgecolor="none")
    ax.axvline(0.3, color="red", ls="--", lw=1, label="near-miss thresh")
    ax.set_xlabel("Clearance (m)")
    ax.set_ylabel("Frame count")
    ax.set_title("Clearance distribution")
    ax.legend(fontsize=8)

    # Traversability
    ax = axes[0, 1]
    tr = agg["traversability"]
    ax.hist(tr, bins=np.arange(tr.min(), tr.max() + 2) - 0.5,
            color="#55A868", edgecolor="none")
    ax.set_xlabel("Traversability (look-ahead steps)")
    ax.set_ylabel("Frame count")
    ax.set_title("Traversability distribution")

    # Beacon range (when visible)
    ax = axes[0, 2]
    br = agg["beacon_range"]
    if len(br) > 0:
        ax.hist(np.clip(br, 0, 15), bins=50, color="#C44E52", edgecolor="none")
        ax.set_xlabel("Beacon range (m) — when visible")
        ax.set_ylabel("Frame count")
        ax.set_title("Beacon range distribution")
    else:
        ax.text(0.5, 0.5, "No beacon-visible frames", ha="center", va="center",
                transform=ax.transAxes)
        ax.set_title("Beacon range distribution")

    # Beacon visibility rate
    ax = axes[1, 0]
    vis_rate = agg["beacon_visible"].mean() * 100
    notvis = 100 - vis_rate
    ax.pie([vis_rate, notvis], labels=[f"Visible\n{vis_rate:.1f}%",
                                        f"Not visible\n{notvis:.1f}%"],
           colors=["#64B5CD", "#CCCCCC"], startangle=90,
           wedgeprops={"edgecolor": "white"})
    ax.set_title("Beacon visibility")

    # Command pattern breakdown
    ax = axes[1, 1]
    cp = agg["cmd_pattern"].astype(int)
    n_patterns = 8
    counts = np.bincount(np.clip(cp, 0, n_patterns - 1), minlength=n_patterns)
    labels = [COMMAND_PATTERN_NAMES[i] if i < len(COMMAND_PATTERN_NAMES) else f"pat_{i}"
              for i in range(n_patterns)]
    colors = plt.cm.tab10(np.linspace(0, 1, n_patterns))
    bars = ax.bar(range(n_patterns), counts, color=colors)
    ax.set_xticks(range(n_patterns))
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=7)
    ax.set_ylabel("Frame count")
    ax.set_title("Command pattern distribution")

    # Collision rate
    ax = axes[1, 2]
    col_rate = agg["collisions"].mean() * 100
    ax.pie([col_rate, 100 - col_rate],
           labels=[f"Collision\n{col_rate:.2f}%", f"Clean\n{100-col_rate:.2f}%"],
           colors=["#E74C3C", "#2ECC71"], startangle=90,
           wedgeprops={"edgecolor": "white"})
    ax.set_title("Collision rate")

    fig.suptitle(f"Label distributions across {len(npz_files)} chunks", fontsize=12)
    plt.tight_layout()
    savefig(fig, os.path.join(out_dir, "03_label_distributions.png"))


# ------------------------------------------------------------------ #
# Figure 4 — Per-chunk summary heatmap
# ------------------------------------------------------------------ #

def fig_chunk_heatmap(npz_files, out_dir):
    if not npz_files:
        return

    metrics = {
        "beacon_vis %":    [],
        "collision %":     [],
        "mean clearance":  [],
        "mean traversab.": [],
        "done rate %":     [],
        "mean beacon rng": [],
    }
    chunk_names = []

    for path in npz_files:
        sample, _ = load_npz_sample(path, n_env_samples=64)
        chunk_names.append(os.path.basename(path).replace(".npz", ""))

        bv = sample["beacon_visible"].astype(float)
        cl = sample["clearance"]
        cl_valid = cl[cl < 900]

        metrics["beacon_vis %"].append(bv.mean() * 100)
        metrics["collision %"].append(sample["collisions"].mean() * 100)
        metrics["mean clearance"].append(cl_valid.mean() if len(cl_valid) else 0)
        metrics["mean traversab."].append(sample["traversability"].mean())
        metrics["done rate %"].append(sample["dones"].mean() * 100)

        br = sample["beacon_range"][sample["beacon_visible"].astype(bool)]
        metrics["mean beacon rng"].append(br.mean() if len(br) else 0)

    # Normalise each metric 0→1 for the heatmap colour
    data = np.array(list(metrics.values()))  # (n_metrics, n_chunks)
    data_norm = np.zeros_like(data)
    for i in range(len(data)):
        mn, mx = data[i].min(), data[i].max()
        data_norm[i] = (data[i] - mn) / (mx - mn + 1e-8)

    fig, ax = plt.subplots(figsize=(max(8, len(npz_files) * 0.7), 5))
    im = ax.imshow(data_norm, aspect="auto", cmap="YlOrRd")

    ax.set_xticks(range(len(chunk_names)))
    ax.set_xticklabels(chunk_names, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(metrics)))
    ax.set_yticklabels(list(metrics.keys()), fontsize=9)

    # Annotate cells with raw values
    for i, (metric, vals) in enumerate(metrics.items()):
        for j, v in enumerate(vals):
            fmt = f"{v:.1f}" if abs(v) < 100 else f"{v:.0f}"
            ax.text(j, i, fmt, ha="center", va="center", fontsize=6.5,
                    color="black" if data_norm[i, j] < 0.6 else "white")

    fig.colorbar(im, ax=ax, shrink=0.6, label="Normalised value")
    ax.set_title("Per-chunk statistics", fontsize=11)
    plt.tight_layout()
    savefig(fig, os.path.join(out_dir, "04_chunk_heatmap.png"))


# ------------------------------------------------------------------ #
# Figure 5 — Vision pixel statistics per chunk
# ------------------------------------------------------------------ #

def fig_vision_stats(h5_files, out_dir):
    if not h5_files:
        return

    chunk_names, means, stds, p5s, p95s = [], [], [], [], []

    for path in h5_files:
        with h5py.File(path, "r") as f:
            n_envs = f["vision"].shape[0]
            T = f["vision"].shape[1]
            # Sample ~16 envs at mid-trajectory
            env_idx = np.linspace(0, n_envs - 1, 16, dtype=int)
            t_mid = T // 2
            frames = f["vision"][env_idx, t_mid].astype(np.float32)  # (16, 3, H, W)

        chunk_names.append(os.path.basename(path).replace("_rgb.h5", ""))
        means.append(frames.mean())
        stds.append(frames.std())
        p5s.append(np.percentile(frames, 5))
        p95s.append(np.percentile(frames, 95))

    x = np.arange(len(chunk_names))
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(max(8, len(h5_files) * 0.8), 6),
                                    sharex=True)

    ax1.plot(x, means, "o-", color="#4C72B0", label="mean pixel")
    ax1.fill_between(x, p5s, p95s, alpha=0.2, color="#4C72B0", label="p5–p95")
    ax1.axhline(128, color="gray", ls="--", lw=0.8, label="mid-range (128)")
    ax1.set_ylabel("Pixel value (0–255)")
    ax1.set_title("Vision pixel statistics per chunk")
    ax1.legend(fontsize=8)
    ax1.set_ylim(0, 255)

    ax2.plot(x, stds, "s-", color="#C44E52", label="pixel std")
    ax2.axhline(30, color="gray", ls="--", lw=0.8, label="min healthy std (30)")
    ax2.set_ylabel("Pixel std dev")
    ax2.set_xlabel("Chunk")
    ax2.legend(fontsize=8)

    ax2.set_xticks(x)
    ax2.set_xticklabels(chunk_names, rotation=45, ha="right", fontsize=8)

    plt.tight_layout()
    savefig(fig, os.path.join(out_dir, "05_vision_stats.png"))


# ------------------------------------------------------------------ #
# Figure 6 — Beacon bearing/range scatter (polar)
# ------------------------------------------------------------------ #

def fig_beacon_polar(npz_files, out_dir, n_chunks=4):
    files = npz_files[:n_chunks]
    if not files:
        return

    ncols = min(len(files), 4)
    nrows = int(np.ceil(len(files) / ncols))
    fig = plt.figure(figsize=(ncols * 3.5, nrows * 3.5))

    for ci, path in enumerate(files):
        ax = fig.add_subplot(nrows, ncols, ci + 1, projection="polar")
        data = np.load(path, allow_pickle=True)
        bv   = data["beacon_visible"].astype(bool)
        bear = data["beacon_bearing"][bv].ravel()
        rng  = data["beacon_range"][bv].ravel()
        data.close()

        if len(bear) == 0:
            ax.set_title(os.path.basename(path).replace(".npz", "") + "\n(no detections)")
            continue

        # Subsample for speed
        idx = np.random.choice(len(bear), min(5000, len(bear)), replace=False)
        ax.scatter(bear[idx], rng[idx], s=1, alpha=0.3, c=rng[idx],
                   cmap="plasma_r", vmin=0, vmax=8)
        ax.set_rmax(8)
        ax.set_title(os.path.basename(path).replace(".npz", ""), fontsize=8, pad=10)
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)

    fig.suptitle("Beacon bearing/range (polar) — when visible", fontsize=11, y=1.01)
    plt.tight_layout()
    savefig(fig, os.path.join(out_dir, "06_beacon_polar.png"))


# ------------------------------------------------------------------ #
# Main
# ------------------------------------------------------------------ #

def main():
    p = argparse.ArgumentParser(description="Generate diagnostic visuals for leWMQuad data")
    p.add_argument("--raw_dir", default=None,
                   help="Directory containing raw .npz chunk files")
    p.add_argument("--h5_dir",  default=None,
                   help="Directory containing rendered *_rgb.h5 chunk files")
    p.add_argument("--out_dir", default="lewm_data_figs",
                   help="Output directory for PNG figures (default: lewm_data_figs)")
    args = p.parse_args()

    if args.raw_dir is None and args.h5_dir is None:
        p.error("Specify at least one of --raw_dir or --h5_dir")

    os.makedirs(args.out_dir, exist_ok=True)
    print(f"Output directory: {args.out_dir}")

    npz_files = sorted(glob.glob(os.path.join(args.raw_dir, "chunk_*.npz"))) \
                if args.raw_dir else []
    h5_files  = sorted(glob.glob(os.path.join(args.h5_dir,  "*_rgb.h5"))) \
                if args.h5_dir else []

    if npz_files:
        print(f"\nFound {len(npz_files)} .npz files")
    if h5_files:
        print(f"Found {len(h5_files)} .h5 files")

    if h5_files:
        print("\n[1/6] RGB frame grid …")
        fig_rgb_grid(h5_files, args.out_dir)

    if npz_files:
        print("\n[2/6] Trajectory plots …")
        fig_trajectories(npz_files, args.out_dir)

        print("\n[3/6] Label distributions …")
        fig_label_distributions(npz_files, args.out_dir)

        print("\n[4/6] Per-chunk heatmap …")
        fig_chunk_heatmap(npz_files, args.out_dir)

    if h5_files:
        print("\n[5/6] Vision pixel stats …")
        fig_vision_stats(h5_files, args.out_dir)

    if npz_files:
        print("\n[6/6] Beacon polar plots …")
        fig_beacon_polar(npz_files, args.out_dir)

    print(f"\nDone. Figures written to {args.out_dir}/")


if __name__ == "__main__":
    main()
