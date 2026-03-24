---
name: leWMQuad project context
description: LeWorldModel conversion of TinyQuadJEPA-v2 quadruped world model — architecture, status, and key decisions
type: project
---

LeWMQuad converts the EMA student-teacher JEPA (TinyQuadJEPA-v2) to the LeWorldModel architecture from Maes et al. (arXiv:2603.19312, Mar 2026).

**Why:** The old CanonicalJEPA used EMA + stop-gradient to prevent collapse (BYOL-style), which is heuristic and fragile. LeWM replaces this with a single encoder + SIGReg regulariser (provable anti-collapse via Cramér-Wold + Epps-Pulley normality test). This also replaces 6 VICReg-style hyperparameters with 1 (λ for SIGReg weight).

**How to apply:**
- The core model lives in `lewm/models/` (sigreg.py, encoders.py, predictor.py, lewm.py)
- Training script: `scripts/3_train_lewm.py`
- Utility files (obstacle_utils, texture_utils, data/, etc.) are copied from TinyQuadJEPA-v2
- The GRU predictor was replaced with a 4-layer Transformer with AdaLN action conditioning
- Projectors with BatchNorm (not LayerNorm) are needed after encoder and predictor because SIGReg tests batch distribution, not per-sample
- The energy head and planning code still need adaptation for the transformer predictor API
- Data generation scripts (1, 2) and evaluation scripts (4-8) still need porting from tqjepa → lewm imports
