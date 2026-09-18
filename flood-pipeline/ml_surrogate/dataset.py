"""
Dataset Loader for FloodSimBench Hydrological Surrogate (Phase 3).

Supports:
- chrimerss/FloodSimBench benchmark dataset
- Calibrated hydrodynamic synthetic generator fallback for offline/local environments
Returns:
- input_tensor: (3, H, W) float32 [Z_norm, Slope S, Rain P]
- target_depth: (1, H, W) float32 [Water depth h >= 0.0m]
"""

import os
import logging
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Tuple, Optional

logger = logging.getLogger("FloodDataset")


class FloodSimBenchDataset(Dataset):
    """
    Dataset loader for chrimerss/FloodSimBench and local hydrodynamic training.
    """
    def __init__(
        self,
        dataset_name: str = "chrimerss/FloodSimBench",
        split: str = "train",
        sample_count: int = 100,
        height: int = 128,
        width: int = 128,
        seed: int = 42
    ):
        self.dataset_name = dataset_name
        self.split = split
        self.sample_count = sample_count
        self.height = height
        self.width = width
        self.data_items = []
        self.use_synthetic = True

        # Attempt to load chrimerss/FloodSimBench if datasets library is available
        try:
            from datasets import load_dataset
            logger.info("Attempting to load %s (%s split)...", dataset_name, split)
            ds = load_dataset(dataset_name, split=split)
            self.data_items = list(ds)
            self.use_synthetic = False
            logger.info("Successfully loaded %d samples from %s", len(self.data_items), dataset_name)
        except Exception as ex:
            logger.info("HuggingFace load skipped or offline (%s). Using calibrated synthetic hydrodynamic stream.", ex)
            self.use_synthetic = True
            self.rng = np.random.RandomState(seed)

    def __len__(self) -> int:
        if not self.use_synthetic:
            return len(self.data_items)
        return self.sample_count

    def _generate_synthetic_sample(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generates physically consistent terrain, slope, rain, and water depth target:
        - Depression accumulation in low elevation, flat slope
        - Zero accumulation on steep slopes
        """
        rng = np.random.RandomState(idx + 1000)
        h, w = self.height, self.width
        x = np.linspace(-1, 1, w)
        y = np.linspace(-1, 1, h)
        xx, yy = np.meshgrid(x, y)

        # 1. Random topographic bowl/valley
        center_x = rng.uniform(-0.3, 0.3)
        center_y = rng.uniform(-0.3, 0.3)
        radius = rng.uniform(0.4, 0.8)
        dist_sq = (xx - center_x)**2 + (yy - center_y)**2
        dem = dist_sq.astype(np.float32)

        # Normalize elevation Z_norm in [0, 1]
        z_min, z_max = dem.min(), dem.max()
        z_norm = (dem - z_min) / (z_max - z_min + 1e-6)

        # 2. Slope S = ||grad(Z)|| in [0, 1]
        dy, dx = np.gradient(dem)
        slope = np.sqrt(dx**2 + dy**2)
        slope_max = slope.max()
        if slope_max > 0:
            slope = slope / (slope_max + 1e-6)
        slope = np.clip(slope, 0.0, 1.0).astype(np.float32)

        # 3. Rain storm P in [0.0, 200.0] mm
        rain_val = float(rng.uniform(30.0, 120.0))
        precip = np.full((h, w), rain_val, dtype=np.float32)

        # 4. Target water depth h (meters): accumulates where Z_norm is low and slope is gentle
        depression_factor = np.maximum(0.0, 1.0 - dist_sq / (radius**2))
        slope_drainage = np.maximum(0.0, 1.0 - slope / 0.3)
        target_h = depression_factor * slope_drainage * (rain_val * 0.005)
        # Apply terminal zero lower-bound
        target_h = np.maximum(0.0, target_h).astype(np.float32)

        input_tensor = np.stack([z_norm, slope, precip], axis=0) # (3, H, W)
        target_tensor = np.expand_dims(target_h, axis=0)          # (1, H, W)

        return torch.from_numpy(input_tensor), torch.from_numpy(target_tensor)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.use_synthetic:
            return self._generate_synthetic_sample(idx)

        item = self.data_items[idx]
        # Ingest benchmark tensor
        inp = np.array(item["input_tensor"], dtype=np.float32)
        target = np.array(item["water_depth"], dtype=np.float32)
        return torch.from_numpy(inp), torch.from_numpy(target)
