#!/usr/bin/env python3
"""Validate raw physics rollout (.npz) and rendered HDF5 (.h5) data chunks.

Usage:
    # Validate both stages:
    python scripts/validate_data.py --raw_dir jepa_raw_data --h5_dir jepa_final_dataset

    # Validate one stage only:
    python scripts/validate_data.py --raw_dir jepa_raw_data
    python scripts/validate_data.py --h5_dir jepa_final_dataset
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import h5py
import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# ------------------------------------------------------------------ #
# Expected schemas
# ------------------------------------------------------------------ #

NPZ_REQUIRED = {
    "proprio":          (None, None, 47),
    "cmds":             (None, None, 3),
    "dones":            (None, None),
    "collisions":       (None, None),
    "base_pos":         (None, None, 3),
    "base_quat":        (None, None, 4),
    "joint_pos":        (None, None, 12),
    "clearance":        (None, None),
    "near_miss":        (None, None),
    "traversability":   (None, None),
    "beacon_visible":   (None, None),
    "beacon_identity":  (None, None),
    "beacon_bearing":   (None, None),
    "beacon_range":     (None, None),
    "cmd_pattern":      (None, None),
}

H5_REQUIRED = {
    "vision":       (None, None, 3, None, None),   # (n_envs, T, 3, H, W)
    "proprio":      (None, None, 47),
    "cmds":         (None, None, 3),
    "dones":        (None, None),
    "collisions":   (None, None),
}

H5_OPTIONAL = {
    "clearance":        (None, None),
    "near_miss":        (None, None),
    "traversability":   (None, None),
    "beacon_visible":   (None, None),
    "beacon_identity":  (None, None),
    "beacon_bearing":   (None, None),
    "beacon_range":     (None, None),
    "cmd_pattern":      (None, None),
}

# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #

RED   = "\033[31m"
GRN   = "\033[32m"
YLW   = "\033[33m"
RST   = "\033[0m"

def ok(msg):   print(f"  {GRN}✓{RST} {msg}")
def warn(msg): print(f"  {YLW}⚠{RST} {msg}")
def fail(msg): print(f"  {RED}✗{RST} {msg}")


def shape_matches(actual, expected):
    """Check shape against a tuple where None means 'any'."""
    if len(actual) != len(expected):
        return False
    return all(e is None or a == e for a, e in zip(actual, expected))


def check_finite(arr, name):
    if np.issubdtype(arr.dtype, np.floating):
        n_bad = np.sum(~np.isfinite(arr))
        if n_bad > 0:
            fail(f"{name}: {n_bad} non-finite values (NaN/Inf)")
            return False
    return True


def sample_envs(n_envs, k=32):
    """Return up to k evenly-spaced env indices."""
    if n_envs <= k:
        return list(range(n_envs))
    step = n_envs // k
    return list(range(0, n_envs, step))[:k]


# ------------------------------------------------------------------ #
# NPZ validation
# ------------------------------------------------------------------ #

def validate_npz(path: str) -> bool:
    name = os.path.basename(path)
    print(f"\n{'─'*60}")
    print(f"NPZ  {name}")
    errors = 0

    try:
        data = np.load(path, allow_pickle=True)
    except Exception as e:
        fail(f"Cannot load: {e}")
        return False

    keys = set(data.files)

    # 1. Required keys and shapes
    for field, expected_shape in NPZ_REQUIRED.items():
        if field not in keys:
            fail(f"Missing field: {field}")
            errors += 1
            continue
        arr = data[field]
        if not shape_matches(arr.shape, expected_shape):
            fail(f"{field}: shape {arr.shape}, expected suffix {expected_shape}")
            errors += 1
        else:
            ok(f"{field}: {arr.shape}  dtype={arr.dtype}")

    # 2. Consistent leading dims across all arrays
    shapes = {f: data[f].shape for f in NPZ_REQUIRED if f in keys}
    leading_dims = {s[:2] for s in shapes.values() if len(s) >= 2}
    if len(leading_dims) > 1:
        fail(f"Inconsistent (n_envs, steps) across fields: {leading_dims}")
        errors += 1
    else:
        n_envs, steps = next(iter(leading_dims))
        ok(f"Consistent dims: n_envs={n_envs}  steps={steps}")

    # 3. JSON metadata
    for meta in ("obstacle_layout", "beacon_layout"):
        if meta in keys:
            try:
                raw = data[meta].item()
                json.loads(raw) if isinstance(raw, str) else raw
                ok(f"{meta}: valid JSON")
            except Exception as e:
                fail(f"{meta}: bad JSON — {e}")
                errors += 1
        else:
            warn(f"{meta}: missing (optional)")

    # Abort deeper checks if leading dims are broken
    if errors > 0 and len(leading_dims) != 1:
        return False

    # 4. Value-range sanity checks (sample envs for speed)
    envs = sample_envs(n_envs)

    clearance = data["clearance"][envs]
    if np.any(clearance < 0):
        fail(f"clearance: {np.sum(clearance < 0)} negative values")
        errors += 1
    else:
        ok(f"clearance range: [{clearance.min():.3f}, {clearance.max():.3f}]")

    bearing = data["beacon_bearing"][envs]
    if not check_finite(bearing, "beacon_bearing"):
        errors += 1
    elif np.any(np.abs(bearing) > np.pi + 0.01):
        fail(f"beacon_bearing out of [-π, π]: max abs = {np.abs(bearing).max():.3f}")
        errors += 1
    else:
        ok(f"beacon_bearing in range")

    b_range = data["beacon_range"][envs]
    if np.any(b_range < 0):
        fail(f"beacon_range: {np.sum(b_range < 0)} negative values")
        errors += 1
    else:
        ok(f"beacon_range range: [{b_range.min():.2f}, {b_range.max():.2f}]")

    # beacon_identity = -1 where not visible
    b_vis  = data["beacon_visible"][envs].astype(bool)
    b_id   = data["beacon_identity"][envs]
    bad = (~b_vis) & (b_id != -1)
    if bad.any():
        fail(f"beacon_identity: {bad.sum()} frames have id != -1 when not visible")
        errors += 1
    else:
        ok("beacon_identity consistent with beacon_visible")

    # cmd_pattern values
    n_patterns = 8  # OU + 7 structured (from command_utils)
    cp = data["cmd_pattern"][envs]
    if np.any(cp < 0) or np.any(cp >= n_patterns):
        fail(f"cmd_pattern: values outside [0, {n_patterns-1}]: {np.unique(cp)}")
        errors += 1
    else:
        ok(f"cmd_pattern values: {np.unique(cp)}")

    # proprio finite
    prop = data["proprio"][envs]
    if not check_finite(prop, "proprio"):
        errors += 1
    else:
        ok(f"proprio finite: mean={prop.mean():.3f}  std={prop.std():.3f}")

    # quaternion norm
    quat = data["base_quat"][envs]
    norms = np.linalg.norm(quat, axis=-1)
    if np.any(np.abs(norms - 1.0) > 0.05):
        fail(f"base_quat: {np.sum(np.abs(norms - 1.0) > 0.05)} badly non-unit quaternions")
        errors += 1
    else:
        ok(f"base_quat norms OK (mean={norms.mean():.4f})")

    data.close()

    if errors == 0:
        print(f"  {GRN}PASS{RST}")
    else:
        print(f"  {RED}FAIL ({errors} errors){RST}")
    return errors == 0


# ------------------------------------------------------------------ #
# HDF5 validation
# ------------------------------------------------------------------ #

def validate_h5(path: str) -> bool:
    name = os.path.basename(path)
    print(f"\n{'─'*60}")
    print(f"H5   {name}")
    errors = 0

    try:
        h5f = h5py.File(path, "r")
    except Exception as e:
        fail(f"Cannot open: {e}")
        return False

    with h5f:
        keys = set(h5f.keys())

        # 1. Required fields
        for field, expected_shape in H5_REQUIRED.items():
            if field not in keys:
                fail(f"Missing required field: {field}")
                errors += 1
                continue
            shape = h5f[field].shape
            if not shape_matches(shape, expected_shape):
                fail(f"{field}: shape {shape}, expected pattern {expected_shape}")
                errors += 1
            else:
                ok(f"{field}: {shape}  dtype={h5f[field].dtype}")

        if errors > 0:
            print(f"  {RED}FAIL ({errors} errors){RST}")
            return False

        n_envs = h5f["vision"].shape[0]
        T      = h5f["vision"].shape[1]
        H      = h5f["vision"].shape[3]
        W      = h5f["vision"].shape[4]

        # 2. Consistent leading dims
        for field in H5_REQUIRED:
            if field in keys:
                s = h5f[field].shape
                if s[0] != n_envs or s[1] != T:
                    fail(f"{field}: leading dims {s[:2]} != ({n_envs}, {T})")
                    errors += 1

        if H != W:
            fail(f"vision: non-square frames ({H}x{W})")
            errors += 1
        else:
            ok(f"Frame resolution: {H}x{W}")

        # 3. Optional label fields
        present_labels = [f for f in H5_OPTIONAL if f in keys]
        missing_labels = [f for f in H5_OPTIONAL if f not in keys]
        ok(f"Label fields present: {present_labels or 'none'}")
        if missing_labels:
            warn(f"Label fields missing: {missing_labels}")

        # 4. Vision pixel statistics (sample envs)
        envs = sample_envs(n_envs)
        # Load a thin temporal slice to avoid reading all frames
        t_mid = T // 2
        vis_sample = h5f["vision"][envs, t_mid]  # (k, 3, H, W) uint8

        if vis_sample.dtype != np.uint8:
            fail(f"vision dtype is {vis_sample.dtype}, expected uint8")
            errors += 1

        mean_val = vis_sample.mean()
        std_val  = vis_sample.std()
        if mean_val < 5:
            fail(f"vision: very dark (mean={mean_val:.1f}) — possible black frames")
            errors += 1
        elif mean_val > 250:
            fail(f"vision: very bright (mean={mean_val:.1f}) — possible white frames")
            errors += 1
        else:
            ok(f"vision pixel stats: mean={mean_val:.1f}  std={std_val:.1f}")

        if std_val < 5:
            fail(f"vision: very low variance (std={std_val:.1f}) — possible collapsed frames")
            errors += 1

        # Check a few all-zero frames
        zero_frames = np.sum(vis_sample.reshape(len(envs), -1).max(axis=1) == 0)
        if zero_frames > 0:
            fail(f"vision: {zero_frames}/{len(envs)} sampled frames are all-zero")
            errors += 1

        # 5. Proprio finite
        prop_sample = h5f["proprio"][envs, t_mid]
        if not check_finite(prop_sample, "proprio"):
            errors += 1
        else:
            ok(f"proprio finite: mean={prop_sample.mean():.3f}  std={prop_sample.std():.3f}")

        # 6. Beacon consistency (if labels present)
        if "beacon_visible" in keys and "beacon_identity" in keys:
            bv = h5f["beacon_visible"][envs, t_mid].astype(bool)
            bi = h5f["beacon_identity"][envs, t_mid]
            bad = (~bv) & (bi != -1)
            if bad.any():
                fail(f"beacon_identity: {bad.sum()} frames id != -1 when not visible")
                errors += 1
            else:
                ok("beacon_identity consistent with beacon_visible")

        # 7. Clearance non-negative
        if "clearance" in keys:
            cl_sample = h5f["clearance"][envs, t_mid]
            if np.any(cl_sample < 0):
                fail(f"clearance: {np.sum(cl_sample < 0)} negative values")
                errors += 1
            else:
                ok(f"clearance range: [{cl_sample.min():.3f}, {cl_sample.max():.3f}]")

    if errors == 0:
        print(f"  {GRN}PASS{RST}")
    else:
        print(f"  {RED}FAIL ({errors} errors){RST}")
    return errors == 0


# ------------------------------------------------------------------ #
# Cross-validation: npz ↔ h5 pairing
# ------------------------------------------------------------------ #

def check_pairing(raw_dir: str, h5_dir: str):
    print(f"\n{'─'*60}")
    print("Pairing check: raw .npz ↔ rendered .h5")
    npz_chunks = {
        os.path.basename(p).replace(".npz", "")
        for p in glob.glob(os.path.join(raw_dir, "chunk_*.npz"))
    }
    h5_chunks = {
        os.path.basename(p).replace("_rgb.h5", "")
        for p in glob.glob(os.path.join(h5_dir, "*_rgb.h5"))
    }
    rendered   = npz_chunks & h5_chunks
    unrendered = npz_chunks - h5_chunks
    orphaned   = h5_chunks - npz_chunks

    ok(f"{len(rendered)} chunk(s) have both .npz and .h5")
    if unrendered:
        warn(f"{len(unrendered)} .npz chunk(s) not yet rendered: {sorted(unrendered)}")
    if orphaned:
        warn(f"{len(orphaned)} .h5 chunk(s) with no matching .npz: {sorted(orphaned)}")


# ------------------------------------------------------------------ #
# Main
# ------------------------------------------------------------------ #

def main():
    p = argparse.ArgumentParser(description="Validate leWMQuad data pipeline outputs")
    p.add_argument("--raw_dir", default=None,
                   help="Directory containing raw .npz chunk files")
    p.add_argument("--h5_dir",  default=None,
                   help="Directory containing rendered *_rgb.h5 chunk files")
    p.add_argument("--chunk",   default=None,
                   help="Validate a single chunk name only (e.g. chunk_0014)")
    args = p.parse_args()

    if args.raw_dir is None and args.h5_dir is None:
        p.error("Specify at least one of --raw_dir or --h5_dir")

    results = []

    if args.raw_dir:
        pattern = os.path.join(args.raw_dir, f"{args.chunk or 'chunk_*'}.npz")
        npz_files = sorted(glob.glob(pattern))
        if not npz_files:
            print(f"{RED}No .npz files found matching: {pattern}{RST}")
        else:
            print(f"\n{'='*60}")
            print(f"RAW NPZ  ({len(npz_files)} files in {args.raw_dir})")
            for path in npz_files:
                results.append(validate_npz(path))

    if args.h5_dir:
        pattern = os.path.join(args.h5_dir, f"{args.chunk or 'chunk_*'}_rgb.h5")
        h5_files = sorted(glob.glob(pattern))
        if not h5_files:
            print(f"{RED}No .h5 files found matching: {pattern}{RST}")
        else:
            print(f"\n{'='*60}")
            print(f"RENDERED H5  ({len(h5_files)} files in {args.h5_dir})")
            for path in h5_files:
                results.append(validate_h5(path))

    if args.raw_dir and args.h5_dir:
        check_pairing(args.raw_dir, args.h5_dir)

    # Summary
    n_pass = sum(results)
    n_fail = len(results) - n_pass
    print(f"\n{'='*60}")
    print(f"Summary: {GRN}{n_pass} passed{RST}  {RED}{n_fail} failed{RST}  "
          f"({len(results)} files checked)")
    sys.exit(0 if n_fail == 0 else 1)


if __name__ == "__main__":
    main()
