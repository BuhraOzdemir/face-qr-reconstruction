import torch.nn as nn
from pytorch_msssim import ssim


class ReconstructionLoss(nn.Module):
    """
    Combined L1 + SSIM loss for sharper reconstruction.

    Both predictions and targets must be in [-1, 1].
    data_range=2.0 accounts for the [-1, 1] value range.
    """

    def __init__(self, ssim_weight: float = 0.5):
        super().__init__()
        self.l1 = nn.L1Loss()
        self.ssim_weight = ssim_weight

    def forward(self, prediction, target):
        l1_loss = self.l1(prediction, target)

        # SSIM expects values in [0, data_range]; data_range=2.0 covers [-1, 1]
        ssim_val = ssim(prediction, target, data_range=2.0, size_average=True)
        ssim_loss = 1.0 - ssim_val

        return l1_loss + self.ssim_weight * ssim_loss
