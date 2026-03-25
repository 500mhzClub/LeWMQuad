# leWMQuad

A quadruped world model built on [LeWorldModel (LeWM)](https://arxiv.org/abs/2603.19312) — a stable, end-to-end Joint-Embedding Predictive Architecture (JEPA) that trains directly from raw pixels using only two loss terms. This project ports the TinyQuadJEPA-v2 quadruped pipeline from EMA-based student-teacher JEPA to the LeWM architecture, replacing heuristic collapse prevention with the theoretically grounded SIGReg regularizer.

## What changed from TinyQuadJEPA-v2

| | TinyQuadJEPA-v2 | leWMQuad |
|---|---|---|
| Collapse prevention | EMA target encoder + stop-gradient | SIGReg (Cramér-Wold normality test) |
| Encoder | Online + Target (EMA updated) | Single encoder, all-gradient |
| Predictor | GRUCell + action conditioning | 6-layer Transformer + AdaLN |
| Projectors | LayerNorm | BatchNorm (required for SIGReg) |
| Tunable loss hyperparameters | 6 (VICReg-style) | 1 (λ for SIGReg weight) |

## Repository layout

```
scripts/
  1_physics_rollout.py      # Genesis simulation → raw .npz chunks
  2_visual_renderer.py      # .npz → rendered HDF5 with domain randomization
  3_train_lewm.py           # LeWM training loop
  validate_data.py          # Validate raw and rendered data
  visualize_data.py         # Diagnostic figures for generated data

lewm/
  models/
    lewm.py                 # LeWorldModel: encoder + predictor + losses
    encoders.py             # ViT-Tiny vision encoder + BatchNorm projector
    predictor.py            # Transformer predictor with AdaLN action conditioning
    sigreg.py               # SIGReg anti-collapse regularizer
    energy_head.py          # Post-hoc goal-compatibility scorer (not yet ported)
    ppo.py                  # Frozen PPO actor-critic (data collection only)
  data/
    streaming_dataset.py    # Lazy HDF5 streaming dataset
  maze_utils.py             # Maze topology generators
  beacon_utils.py           # Beacon panel placement and rendering
  label_utils.py            # Clearance / traversability / beacon label computation
  command_utils.py          # OU process + structured command patterns
  obstacle_utils.py         # Random obstacle layout generation
  texture_utils.py          # Procedural ground texture generation

assets/
  mini_pupper/              # Robot URDF and meshes
```

## Data pipeline

Data is produced in two stages before training.

### Stage 1 — Physics rollout

Runs a frozen PPO blind walking policy in [Genesis](https://genesis-world.readthedocs.io) across 2048 parallel environments. Each chunk captures 1000 timesteps per environment.

```bash
python scripts/1_physics_rollout.py \
    --policy_path checkpoints/ppo_blind.pt \
    --out_dir jepa_raw_data \
    --n_chunks 20
```

Each chunk produces a `.npz` file containing:

| Array | Shape | Description |
|---|---|---|
| `proprio` | (2048, 1000, 47) | Noisy proprioceptive observation |
| `cmds` | (2048, 1000, 3) | Velocity commands [vx, vy, wz] |
| `base_pos/quat` | (2048, 1000, 3/4) | Ground-truth kinematics |
| `joint_pos` | (2048, 1000, 12) | Actuated joint positions |
| `clearance` | (2048, 1000) | Min distance to nearest obstacle |
| `traversability` | (2048, 1000) | Forward look-ahead steps clear |
| `beacon_visible` | (2048, 1000) | Any beacon in camera FOV |
| `beacon_identity` | (2048, 1000) | Closest visible beacon index (−1 = none) |
| `beacon_bearing` | (2048, 1000) | Relative bearing to closest beacon (rad) |
| `beacon_range` | (2048, 1000) | Distance to closest beacon (m) |
| `cmd_pattern` | (2048, 1000) | Command pattern index (0=OU, 1–7=structured) |

**Command patterns:** 50% OU exploration, 50% split across retreat, stop, recovery, dead-end backout, wall-following, and spin-in-place.

**Scene composition per chunk:** random maze topology (T-junction, crossroads, S-bend, zigzag, one/two-turn, multi-room, branching, cul-de-sac, corridor, doorway) + free obstacles + beacon panels + distractor patches.

### Stage 2 — Visual rendering

Replays the recorded kinematics in isolated render scenes, applies domain randomization, and writes HDF5 files.

```bash
python scripts/2_visual_renderer.py \
    --raw_dir jepa_raw_data \
    --out_dir jepa_final_dataset \
    --workers 4
```

Each `chunk_XXXX_rgb.h5` adds:

| Dataset | Shape | Description |
|---|---|---|
| `vision` | (2048, 1000, 3, 224, 224) | uint8 egocentric RGB |

Domain randomization applied per chunk:
- 27 procedural ground textures
- Per-frame brightness ±0.4, contrast 0.5–1.5×, Gaussian noise σ=0.02–0.08, hue shift ±0.08 rad
- Camera pose jitter: 8mm translation, 12mm lookat
- 30% of wall panels get beacon-like colors (hard negatives for beacon recognition)

### Data validation and visualization

```bash
# Validate both stages (exits 1 on any failure)
python scripts/validate_data.py --raw_dir jepa_raw_data --h5_dir jepa_final_dataset

# Single chunk
python scripts/validate_data.py --h5_dir jepa_final_dataset --chunk chunk_0005

# Diagnostic figures
python scripts/visualize_data.py --raw_dir jepa_raw_data --h5_dir jepa_final_dataset
```

## Training

```bash
python scripts/3_train_lewm.py \
    --data_dir jepa_final_dataset \
    --epochs 20 \
    --batch_size 128 \
    --sigreg_lambda 0.1
```

Key arguments:

| Argument | Default | Description |
|---|---|---|
| `--sigreg_lambda` | 0.1 | SIGReg weight λ — the only effective hyperparameter |
| `--seq_len` | 4 | Timesteps per training sequence |
| `--lr` | 2e-3 | Peak learning rate (cosine annealing) |
| `--use_proprio` | off | Fuse proprioception into the encoder |
| `--resume_from` | — | Resume from a checkpoint path |

Checkpoints are saved to `lewm_checkpoints/` every `--save_every` steps and at each epoch end. Training metrics (loss, pred_loss, sigreg_loss, z_proj_std, grad_norm) are logged to `lewm_logs/training_metrics.csv`.

**Collapse monitoring:** a warning is printed when `z_proj_std < 0.1`. The SIGReg loss should drop sharply in early training then plateau near zero; if it stays high, increase λ.

## Model architecture

```
Observation (3, 224, 224)
      │
  ViT-Tiny encoder          ~5M params
  (12 layers, 3 heads,
   192-dim, patch=14)
      │  [CLS] token
  BatchNorm projector        z_t ∈ ℝ¹⁹²
      │
  TransformerPredictor       ~10M params
  (6 layers, 16 heads,
   384 hidden, AdaLN
   action conditioning)
      │
  BatchNorm projector        ẑ_{t+1} ∈ ℝ¹⁹²
```

**Loss:** `L = MSE(ẑ_{t+1}, z_{t+1}) + λ · SIGReg(Z)`

SIGReg projects the batch of latents onto 1024 random unit-norm directions and applies the Epps-Pulley normality test to each projection. By the Cramér-Wold theorem, matching all 1D marginals to N(0,1) is equivalent to matching the full joint distribution to an isotropic Gaussian, preventing collapse without stop-gradient or EMA.

## Planning (evaluation)

Planning reuses the eval scripts from TinyQuadJEPA-v2 (not yet ported). The model exposes:

```python
model.plan_rollout(z_start_raw, action_seq)  # autoregressive latent rollout
model.plan_cost(z_pred_proj, z_goal_proj)    # L2² goal-matching cost
```

These are designed to plug into a CEM solver (512 candidates × 15-step horizon × 4–5 iterations).

## Known limitations

- **Wall height vs. camera height:** maze walls default to 0.15–0.40m, close to the camera height (~0.20m). The encoder sees mostly open floor rather than enclosed corridors. Increasing `wall_height_range` in `generate_maze` will produce more visually enclosed scenes.
- **Teacher-forced training:** the predictor is trained with ground-truth encoded latents at every step. At planning time it runs autoregressively — prediction errors compound over the 15-step rollout horizon (exposure bias). This is a known limitation of the LeWM architecture.
- **Beacon coverage:** beacon panels are currently seen in ~11% of frames and are absent entirely from some chunks depending on maze layout and robot trajectory. Multi-view landmark coverage for latent breadcrumb quality may require denser beacon placement or targeted approach episodes.
- **Energy head and eval scripts** (scripts 4–8 from TinyQuadJEPA-v2) are not yet ported to leWM.

## Reference

Maes, Le Lidec, Scieur, LeCun, Balestriero.
*LeWorldModel: Stable End-to-End Joint-Embedding Predictive Architecture from Pixels.*
arXiv:2603.19312, March 2026.
