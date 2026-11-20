"""Loss functions: soft Dice + binary cross-entropy (the nnU-Net-style combination)."""
import torch
import torch.nn as nn


class SoftDiceLoss(nn.Module):
    """1 - (2*sum(p*g) + s) / (sum(p) + sum(g) + s), computed over the whole batch.
    Batch-level (not per-slice) Dice is more stable because many slices contain no femur.
    """

    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, target):
        prob = torch.sigmoid(logits)
        inter = (prob * target).sum()
        denom = prob.sum() + target.sum()
        return 1.0 - (2.0 * inter + self.smooth) / (denom + self.smooth)


class DiceBCELoss(nn.Module):
    def __init__(self, dice_weight=1.0, bce_weight=1.0):
        super().__init__()
        self.dice = SoftDiceLoss()
        self.bce = nn.BCEWithLogitsLoss()
        self.dw, self.bw = dice_weight, bce_weight

    def forward(self, logits, target):
        logits = logits.float()  # compute the loss in fp32 even under autocast
        return self.dw * self.dice(logits, target) + self.bw * self.bce(logits, target)
