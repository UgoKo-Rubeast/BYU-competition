import torch
import torch.nn as nn
import torch.nn.functional as F


class SmoothBCE(nn.Module):
    def __init__(self, smooth=0.0, pos_weight=None):
        super().__init__()
        if not (0 <= smooth < 1):
            raise ValueError("smooth must be in [0, 1)")
        self.smooth = float(smooth)

        if pos_weight is not None:
            pos_weight_tensor = torch.tensor([float(pos_weight)], dtype=torch.float32)
            self.register_buffer("pos_weight", pos_weight_tensor, persistent=False)
        else:
            self.pos_weight = None

    def forward(self, logits, target):
        target = target.float()
        if self.smooth > 0:
            target = target * (1.0 - self.smooth) + (1.0 - target) * self.smooth
        return F.binary_cross_entropy_with_logits(
            logits,
            target,
            reduction="mean",
            pos_weight=self.pos_weight,
        )


class DiceLoss(nn.Module):
    """Binary Dice loss for 3D segmentation logits."""

    def __init__(self, smooth=1.0, eps=1e-7):
        super().__init__()
        self.smooth = float(smooth)
        self.eps = float(eps)

    def forward(self, logits, target):
        probs = torch.sigmoid(logits)
        target = target.float()

        dims = tuple(range(2, probs.ndim))
        intersection = (probs * target).sum(dim=dims)
        denom = probs.sum(dim=dims) + target.sum(dim=dims)
        dice = (2.0 * intersection + self.smooth) / (denom + self.smooth + self.eps)
        return 1.0 - dice.mean()


class BCEDiceLoss(nn.Module):
    def __init__(self, bce_weight=0.5, dice_weight=0.5, smooth=0.0):
        super().__init__()
        self.bce_weight = float(bce_weight)
        self.dice_weight = float(dice_weight)
        self.bce = SmoothBCE(smooth=smooth)
        self.dice = DiceLoss()

    def forward(self, logits, target):
        return self.bce_weight * self.bce(logits, target) + self.dice_weight * self.dice(logits, target)


def run_section_6_1_assertions():
    torch.manual_seed(123)

    # 4-1 validation A: Dice loss should be near zero on near-perfect prediction.
    dice = DiceLoss(smooth=1.0)
    target = torch.ones(2, 1, 8, 8, 8)
    logits_perfect = torch.full_like(target, 12.0)
    loss_perfect = dice(logits_perfect, target)
    assert loss_perfect.item() < 1e-3

    # 4-1 validation B: Dice loss should be high on opposite prediction.
    logits_wrong = torch.full_like(target, -12.0)
    loss_wrong = dice(logits_wrong, target)
    assert loss_wrong.item() > 0.9

    # 4-1 validation C: Combined custom loss supports backward.
    combo = BCEDiceLoss(bce_weight=0.6, dice_weight=0.4, smooth=0.01)
    logits = torch.randn(2, 1, 8, 8, 8, requires_grad=True)
    target_rand = (torch.rand(2, 1, 8, 8, 8) > 0.6).float()
    loss = combo(logits, target_rand)
    loss.backward()
    assert torch.isfinite(loss).item()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all().item()

    print("[4-1/dice_perfect]", float(loss_perfect))
    print("[4-1/dice_wrong]", float(loss_wrong))
    print("[4-1/bce_dice]", float(loss))
