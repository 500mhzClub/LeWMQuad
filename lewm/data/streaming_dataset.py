"""Streaming HDF5 dataset for JEPA and energy-head training."""
from __future__ import annotations

import glob
import math
import os
from typing import Iterator, List, Tuple

import h5py
import numpy as np
import torch
from torch.utils.data import IterableDataset


class StreamingJEPADataset(IterableDataset):
    """Yields (vision, proprio, cmds, dones, collisions) sequence tuples.

    Streams directly from HDF5 files — no full-dataset RAM copy.  Sequence
    indices are pre-built at init and sharded across all workers by index
    (not by file), so all num_workers are active regardless of file count.

    Args:
        data_dir: directory containing rendered chunk files (``*_rgb.h5``).
        seq_len: number of timesteps per returned sequence.
        batch_size: worker-side micro-batch size.
        require_no_done: skip sequences that contain a ``done`` flag.
        require_no_collision: skip sequences that contain a ``collision`` flag.
    """

    def __init__(
        self,
        data_dir: str,
        seq_len: int = 16,
        batch_size: int = 256,
        require_no_done: bool = True,
        require_no_collision: bool = True,
        num_workers: int = 1,
    ):
        super().__init__()
        self.files: List[str] = self._discover_files(data_dir)
        if not self.files:
            raise FileNotFoundError(f"No rendered HDF5 chunk files found in {data_dir}")
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.require_no_done = require_no_done
        self.require_no_collision = require_no_collision
        self._num_workers = max(1, num_workers)

        # Pre-build index table: list of (file_path, env_idx, t0)
        # Scans dones/collisions at init (small arrays, ~10 MB total).
        self._all_indices: List[Tuple[str, int, int]] = self._precompute_indices()

    @staticmethod
    def _discover_files(data_dir: str) -> List[str]:
        patterns = ("*_rgb.h5", "chunk_*.h5")
        files = []
        seen = set()
        for pattern in patterns:
            for path in sorted(glob.glob(os.path.join(data_dir, pattern))):
                if "_tmp_" in os.path.basename(path):
                    continue
                if path not in seen:
                    files.append(path)
                    seen.add(path)
        return files

    def _precompute_indices(self) -> List[Tuple[str, int, int]]:
        indices = []
        for fpath in self.files:
            with h5py.File(fpath, "r") as h5f:
                n_envs, T = h5f["vision"].shape[:2]
                dones = (
                    h5f["dones"][:] if ("dones" in h5f and self.require_no_done) else None
                )
                collisions = (
                    h5f["collisions"][:]
                    if ("collisions" in h5f and self.require_no_collision)
                    else None
                )
                for e in range(n_envs):
                    for t0 in range(0, T - self.seq_len + 1, self.seq_len):
                        t1 = t0 + self.seq_len
                        if dones is not None and np.any(dones[e, t0:t1]):
                            continue
                        if collisions is not None and np.any(collisions[e, t0:t1]):
                            continue
                        indices.append((fpath, e, t0))
        return indices

    def __len__(self) -> int:
        # Sum per-worker batch counts (each worker rounds up independently)
        n = len(self._all_indices)
        worker_sizes = [len(range(w, n, self._num_workers)) for w in range(self._num_workers)]
        return sum(math.ceil(s / self.batch_size) for s in worker_sizes)

    def __iter__(self) -> Iterator:
        info = torch.utils.data.get_worker_info()

        # Shuffle the full index list each epoch
        rng = np.random.RandomState()
        indices = list(self._all_indices)
        rng.shuffle(indices)

        # Shard by worker index (stride pattern keeps batches balanced)
        if info is not None:
            indices = indices[info.id :: info.num_workers]

        # Keep file handles open for the lifetime of this worker's iteration
        open_files: dict = {}
        try:
            for b0 in range(0, len(indices), self.batch_size):
                batch_idx = indices[b0 : b0 + self.batch_size]
                B = len(batch_idx)
                if B == 0:
                    continue

                vis = np.empty((B, self.seq_len, 3, 64, 64), dtype=np.uint8)
                prop = np.empty((B, self.seq_len, 47), dtype=np.float32)
                cmds = np.empty((B, self.seq_len, 3), dtype=np.float32)
                dones = np.zeros((B, self.seq_len), dtype=np.bool_)
                collisions = np.zeros((B, self.seq_len), dtype=np.bool_)

                for i, (fpath, e, t0) in enumerate(batch_idx):
                    if fpath not in open_files:
                        open_files[fpath] = h5py.File(fpath, "r")
                    h5f = open_files[fpath]
                    t1 = t0 + self.seq_len
                    vis[i] = h5f["vision"][e, t0:t1]
                    prop[i] = h5f["proprio"][e, t0:t1]
                    cmds[i] = h5f["cmds"][e, t0:t1]
                    if "dones" in h5f:
                        dones[i] = h5f["dones"][e, t0:t1]
                    if "collisions" in h5f:
                        collisions[i] = h5f["collisions"][e, t0:t1]

                yield (
                    torch.from_numpy(vis),
                    torch.from_numpy(prop),
                    torch.from_numpy(cmds),
                    torch.from_numpy(dones),
                    torch.from_numpy(collisions),
                )
        finally:
            for f in open_files.values():
                f.close()
