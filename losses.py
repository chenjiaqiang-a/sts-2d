import torch
from torch import nn
import segmentation_models_pytorch as smp

__all__ = ['DiceCELoss']


class DiceCELoss(nn.Module):
    def __init__(self, mode: str = 'binary', lamb_dice: float = 1.0, lamb_ce: float = 1.0):
        """Dice Loss and Cross Entropy Loss

        Args
            - **mode**: Loss mode 'binary', 'multiclass' or 'multilabel'
            - **lamb_dice**: Factor of dice loss
            - **lamb_ce**: Factor of cross entropy loss

        Shape
            - **y_pred**: torch.Tensor of shape (N, C, H, W)
            - **y_true**: torch.Tensor of shape (N, H, W) or (N, C, H, W)
        """
        super(DiceCELoss, self).__init__()
        self.lamb_dice = lamb_dice
        self.lamb_ce = lamb_ce
        self.dice = smp.losses.DiceLoss(mode=mode)
        if mode == 'binary':
            self.ce = smp.losses.SoftBCEWithLogitsLoss()
        else:
            self.ce = smp.losses.SoftCrossEntropyLoss()

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        dice_loss = self.dice(y_pred, y_true)
        ce_loss = self.ce(y_pred, y_true)
        return self.lamb_dice * dice_loss + self.lamb_ce * ce_loss
