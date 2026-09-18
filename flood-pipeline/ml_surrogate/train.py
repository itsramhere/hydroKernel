"""
Training Pipeline for FloodUNet Hydrological Surrogate (Phase 3).

Loss Formulation:
L_total = 1.0 * L_MSE + 0.5 * L_SoftDice + 0.1 * L_Depression
where:
- L_SoftDice uses sigma(10.0 * (y_pred - 0.15)) against (y_true >= 0.15)
- L_Depression = mean(y_pred * I(S > 0.3))
Optimizer: AdamW(lr=1e-3, weight_decay=1e-4) with CosineAnnealingLR
Weights output: flood-pipeline/ml_surrogate/models/flood_unet.pth
"""

import os
import sys
import logging
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Tuple

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from unet import FloodUNet
from dataset import FloodSimBenchDataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("TrainFloodUNet")

MODELS_DIR = os.path.join(CURRENT_DIR, "models")
PTH_PATH = os.path.join(MODELS_DIR, "flood_unet.pth")


class CompoundHydrodynamicLoss(nn.Module):
    """
    Compound Hydrodynamic Loss Function:
    L_total = 1.0 * L_MSE + 0.5 * L_SoftDice + 0.1 * L_Depression
    """
    def __init__(self, depth_threshold: float = 0.15, slope_penalty_thresh: float = 0.3):
        super().__init__()
        self.mse = nn.MSELoss()
        self.depth_threshold = depth_threshold
        self.slope_thresh = slope_penalty_thresh

    def forward(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        input_tensor: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # 1. MSE Loss
        l_mse = self.mse(y_pred, y_true)

        # 2. Soft Dice Loss on inundation classification boundary (h >= 0.15m)
        pred_prob = torch.sigmoid(10.0 * (y_pred - self.depth_threshold))
        true_prob = (y_true >= self.depth_threshold).float()
        intersection = torch.sum(pred_prob * true_prob)
        cardinality = torch.sum(pred_prob) + torch.sum(true_prob)
        dice_score = (2.0 * intersection + 1e-6) / (cardinality + 1e-6)
        l_soft_dice = 1.0 - dice_score

        # 3. Depression Loss: penalize water accumulating on steep slopes (S > 0.3)
        # Channel 1 of input_tensor is Topographic Slope Gradient S
        slope = input_tensor[:, 1:2, :, :]
        steep_mask = (slope > self.slope_thresh).float()
        l_depression = torch.mean(y_pred * steep_mask)

        # Total Weighted Loss
        l_total = 1.0 * l_mse + 0.5 * l_soft_dice + 0.1 * l_depression
        return l_total, l_mse, l_soft_dice, l_depression


def train_surrogate(
    epochs: int = 5,
    batch_size: int = 4,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    device: str = "cpu",
    output_path: str = PTH_PATH
) -> str:
    """Trains FloodUNet and exports PyTorch checkpoint."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    logger.info("Initializing FloodUNet training on %s (epochs=%d, lr=%.4f)...", device, epochs, lr)

    dataset = FloodSimBenchDataset(sample_count=40, height=64, width=64)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = FloodUNet(in_channels=3, out_channels=1).to(device)
    criterion = CompoundHydrodynamicLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    model.train()
    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            pred_depth = model(batch_x)
            loss, l_mse, l_dice, l_dep = criterion(pred_depth, batch_y, batch_x)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * batch_x.size(0)

        scheduler.step()
        avg_loss = epoch_loss / len(dataset)
        logger.info("Epoch [%d/%d] - Loss: %.4f (lr: %.6f)", epoch, epochs, avg_loss, scheduler.get_last_lr()[0])

    torch.save(model.state_dict(), output_path)
    logger.info("Saved trained FloodUNet model weights to %s", output_path)
    return output_path


if __name__ == "__main__":
    train_surrogate(epochs=3)
