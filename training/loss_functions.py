import torch
import torch.nn as nn
import torch.nn.functional as F

class GeometryConsistencyLoss(nn.Module):
    def __init__(self, temperature=8.0):
        super().__init__()
        self.temperature = temperature
        self.loss = nn.SmoothL1Loss()

    def forward(self, pred_mask, pred_sdt):
        sdt_mask = torch.sigmoid(self.temperature * pred_sdt)
        return self.loss(pred_mask, sdt_mask)
    
    

def dice_loss(pred, target, eps=1e-6):
    """Per-sample Dice, averaged over the batch."""
    pred = pred.flatten(1)      # (B, H*W)
    target = target.flatten(1)  # (B, H*W)
    intersection = (pred * target).sum(dim=1)
    union = pred.sum(dim=1) + target.sum(dim=1)
    dice = (2. * intersection + eps) / (union + eps)
    return 1 - dice.mean()

def boundary_weighted_bce(pred, target, boundary, weight=5.0):
    base_loss = F.binary_cross_entropy(pred, target, reduction="none")
    boundary_w = 1.0 + (weight - 1.0) * boundary
    return (base_loss * boundary_w).mean()

def prior_loss(pred, target):
    return F.smooth_l1_loss(pred, target)

consistency_loss = GeometryConsistencyLoss(temperature=8.0)

def total_loss(outputs, batch):

    mask_loss = (
        dice_loss(
            outputs["mask"],
            batch["mask"]
        )
        +
        boundary_weighted_bce(
            outputs["mask"],
            batch["mask"],
            batch["boundary"]
        )
    )

    spatial_loss = prior_loss(
        outputs["prior"],
        batch["distance"]
    )

    boundary_loss = F.binary_cross_entropy(
        outputs["boundary"],
        batch["boundary"]
    )

    loss_consistency = consistency_loss(
        outputs["mask"],
        outputs["prior"]
    )

   

    loss = (
        0.65 * mask_loss +
        0.15 * spatial_loss +
        0.10 * boundary_loss+
        0.10 * loss_consistency
    )

    print(
        f"Mask Loss:{mask_loss:.4f} "
        f"Prior Loss:{spatial_loss:.4f} "
        f"Boundary Loss:{boundary_loss:.4f} "
        f"Consistency Loss:{loss_consistency:.4f}"
    )

    return loss
