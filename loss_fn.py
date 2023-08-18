import torch
from torch import nn

try:
    import segmentation_models_pytorch as m
except ImportError:
    import models as m

__all__ = ['DiceCELoss', 'DiceMSELoss', 'DiceFocalLoss', 'get_loss_fn']


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
        self.dice = m.losses.DiceLoss(mode=mode)
        if mode == 'binary':
            self.ce = m.losses.SoftBCEWithLogitsLoss()
        else:
            self.ce = m.losses.SoftCrossEntropyLoss()

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        dice_loss = self.dice(y_pred, y_true)
        ce_loss = self.ce(y_pred, y_true)
        return self.lamb_dice * dice_loss + self.lamb_ce * ce_loss


class DiceMSELoss(nn.Module):
    def __init__(self, mode: str = 'binary', lamb_dice: float = 1.0, lamb_mse: float = 1.0):
        """Dice Loss and Cross Entropy Loss

        Args
            - **mode**: Loss mode 'binary', 'multiclass' or 'multilabel'
            - **lamb_dice**: Factor of dice loss
            - **lamb_mse**: Factor of mse loss

        Shape
            - **y_pred**: torch.Tensor of shape (N, C, H, W)
            - **y_true**: torch.Tensor of shape (N, H, W) or (N, C, H, W)
        """
        super(DiceMSELoss, self).__init__()
        self.lamb_dice = lamb_dice
        self.lamb_mse = lamb_mse
        self.dice = m.losses.DiceLoss(mode=mode)
        self.mse = nn.MSELoss()
        self.mode = mode

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        dice_loss = self.dice(y_pred, y_true)
        if self.mode == 'binary':
            y_pred = y_pred.squeeze(dim=1)
        mse_loss = self.mse(y_pred, y_true)
        return self.lamb_dice * dice_loss + self.lamb_mse * mse_loss


class DiceFocalLoss(nn.Module):
    def __init__(self, mode: str = 'binary', lamb_dice: float = 1.0, lamb_focal: float = 1.0):
        """Dice Loss and Focal Loss

                Args
                    - **mode**: Loss mode 'binary', 'multiclass' or 'multilabel'
                    - **lamb_dice**: Factor of dice loss
                    - **lamb_ce**: Factor of focal loss

                Shape
                    - **y_pred**: torch.Tensor of shape (N, C, H, W)
                    - **y_true**: torch.Tensor of shape (N, H, W) or (N, C, H, W)
                """
        super(DiceFocalLoss, self).__init__()
        self.lamb_dice = lamb_dice
        self.lamb_focal = lamb_focal
        self.dice = m.losses.DiceLoss(mode=mode)
        self.focal = m.losses.FocalLoss(mode=mode)

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        dice_loss = self.dice(y_pred, y_true)
        focal_loss = self.focal(y_pred, y_true)
        return self.lamb_dice * dice_loss + self.lamb_focal * focal_loss


loss_map = {
    'dice_loss': m.losses.DiceLoss,
    'mse_loss': nn.MSELoss,
    'dice_ce_loss': DiceCELoss,
    'dice_mse_loss': DiceMSELoss,
    'dice_focal_loss': DiceFocalLoss,
}


def get_loss_fn(loss_fn: str = 'dice_ce_loss', **kwargs) -> nn.Module:
    return loss_map[loss_fn](**kwargs)
