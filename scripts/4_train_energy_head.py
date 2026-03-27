#!/usr/bin/env python3
"""Train the LatentEnergyHead probe on frozen LeWM encoder latents.

The head learns a composite energy from clearance, traversability, and
beacon_range labels — giving useful gradient everywhere in the maze for
safe exploration, with beacon attraction layered on top.

Usage:
    python scripts/4_train_energy_head.py \
        --data_dir jepa_final_dataset_224 \
        --checkpoint lewm_checkpoints/epoch_20.pt \
        --epochs 10 --batch_size 256 --lr 3e-4
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import time

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.amp import autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from lewm.models import LeWorldModel, LatentEnergyHead, composite_energy_target
from lewm.data import StreamingJEPADataset
from lewm.checkpoint_utils import clean_state_dict


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train LatentEnergyHead probe")
    p.add_argument("--data_dir", type=str, default="jepa_final_dataset")
    p.add_argument("--checkpoint", type=str, required=True,
                   help="Path to trained LeWM checkpoint.")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--seq_len", type=int, default=4)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--save_every", type=int, default=2000)
    p.add_argument("--out_dir", type=str, default="energy_head_checkpoints")
    p.add_argument("--log_dir", type=str, default="energy_head_logs")
    # Model dims (must match the checkpoint)
    p.add_argument("--latent_dim", type=int, default=192)
    p.add_argument("--hidden_dim", type=int, default=512)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--image_size", type=int, default=None)
    p.add_argument("--patch_size", type=int, default=None)
    p.add_argument("--use_proprio", action="store_true")
    # Composite target weights
    p.add_argument("--w_safety", type=float, default=0.5)
    p.add_argument("--w_mobility", type=float, default=0.3)
    p.add_argument("--w_beacon", type=float, default=0.2)
    p.add_argument("--clearance_clip", type=float, default=1.0,
                   help="Saturate clearance beyond this distance (metres).")
    p.add_argument("--beacon_clip", type=float, default=5.0,
                   help="Saturate beacon_range beyond this distance (metres).")
    return p.parse_args()


def load_frozen_encoder(args, device):
    """Load the LeWM encoder from a checkpoint and freeze it."""
    ckpt = torch.load(args.checkpoint, map_location=device)
    sd = clean_state_dict(ckpt["model_state_dict"])

    image_size = args.image_size or 224
    patch_size = args.patch_size or (14 if image_size == 224 else 4)

    model = LeWorldModel(
        latent_dim=args.latent_dim,
        image_size=image_size,
        patch_size=patch_size,
        use_proprio=args.use_proprio,
    )
    model.load_state_dict(sd, strict=False)
    model = model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def train(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Training LatentEnergyHead on {device}")
    print(f"Target weights: safety={args.w_safety}, mobility={args.w_mobility}, beacon={args.w_beacon}")

    # Dataset
    num_workers = 12
    dataset = StreamingJEPADataset(
        data_dir=args.data_dir,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        require_no_done=False,
        require_no_collision=False,
        num_workers=num_workers,
        load_labels=True,
    )
    channels, height, width = dataset.vision_shape
    if args.image_size is None:
        args.image_size = height
    if args.patch_size is None:
        args.patch_size = 14 if height == 224 else 4

    dataloader = DataLoader(
        dataset, batch_size=None, num_workers=num_workers,
        pin_memory=True, prefetch_factor=2,
    )

    # Frozen encoder
    encoder_model = load_frozen_encoder(args, device)
    print(f"Loaded frozen encoder from {args.checkpoint}")

    # Energy head
    head = LatentEnergyHead(
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    ).to(device)
    head = torch.compile(head)

    n_params = sum(p.numel() for p in head.parameters())
    print(f"Energy head parameters: {n_params:,}")

    optimizer = torch.optim.AdamW(
        head.parameters(), lr=args.lr, weight_decay=args.weight_decay,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Logging
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    csv_path = os.path.join(args.log_dir, "energy_head_metrics.csv")
    if not os.path.exists(csv_path):
        with open(csv_path, mode="w", newline="") as f:
            csv.writer(f).writerow(["step", "epoch", "loss", "mean_energy", "lr"])

    global_step = 0

    for epoch in range(args.epochs):
        head.train()
        epoch_loss_sum = 0.0
        epoch_batches = 0
        t0 = time.time()

        pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{args.epochs}")
        for batch in pbar:
            vision, proprio, cmds, dones, collisions, labels = batch

            clearance = labels.get("clearance")
            traversability = labels.get("traversability")
            beacon_range = labels.get("beacon_range")
            if clearance is None or traversability is None:
                continue

            vision = vision.to(device, non_blocking=True).float().div_(255.0)
            proprio = proprio.to(device, non_blocking=True)
            clearance = clearance.to(device, non_blocking=True).float()
            traversability = traversability.to(device, non_blocking=True)
            beacon_range = beacon_range.to(device, non_blocking=True).float() if beacon_range is not None else torch.full_like(clearance, 999.0)

            # Composite target
            target = composite_energy_target(
                clearance, traversability, beacon_range,
                clearance_clip=args.clearance_clip,
                traversability_horizon=10,
                beacon_clip=args.beacon_clip,
                w_safety=args.w_safety,
                w_mobility=args.w_mobility,
                w_beacon=args.w_beacon,
            )

            B, T = vision.shape[:2]

            # Encode all frames (frozen)
            with torch.no_grad(), autocast("cuda", dtype=torch.bfloat16):
                _, z_proj = encoder_model.encode_seq(vision, proprio)

            # Flatten (B, T) -> (B*T,)
            z_flat = z_proj.reshape(B * T, -1).float()
            target_flat = target.reshape(B * T)

            optimizer.zero_grad(set_to_none=True)

            energy = head(z_flat)
            loss = nn.functional.mse_loss(energy, target_flat)

            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                head.parameters(), max_norm=args.grad_clip,
            ).item()

            if not torch.isfinite(loss) or not math.isfinite(grad_norm):
                optimizer.zero_grad(set_to_none=True)
                continue

            optimizer.step()

            global_step += 1
            loss_val = loss.item()
            mean_energy = energy.detach().mean().item()
            epoch_loss_sum += loss_val
            epoch_batches += 1

            with open(csv_path, mode="a", newline="") as f:
                csv.writer(f).writerow([
                    global_step, epoch + 1,
                    f"{loss_val:.6f}",
                    f"{mean_energy:.4f}",
                    f"{optimizer.param_groups[0]['lr']:.2e}",
                ])

            if global_step % 5 == 0:
                pbar.set_postfix(
                    loss=f"{loss_val:.4f}",
                    energy=f"{mean_energy:.3f}",
                    lr=f"{optimizer.param_groups[0]['lr']:.1e}",
                )

            if global_step % args.save_every == 0:
                ckpt_path = os.path.join(args.out_dir, f"energy_step_{global_step}.pt")
                torch.save({"head_state_dict": head.state_dict(), "step": global_step, "epoch": epoch}, ckpt_path)
                print(f"\n  Saved: {ckpt_path}")

        scheduler.step()
        avg_loss = epoch_loss_sum / max(1, epoch_batches)
        elapsed = time.time() - t0
        print(f"Epoch {epoch + 1} complete | avg_loss={avg_loss:.4f} | time={elapsed:.0f}s")

        epoch_path = os.path.join(args.out_dir, f"energy_epoch_{epoch + 1}.pt")
        torch.save({"head_state_dict": head.state_dict(), "step": global_step, "epoch": epoch}, epoch_path)

    print("Energy head training complete.")


if __name__ == "__main__":
    args = parse_args()
    train(args)
