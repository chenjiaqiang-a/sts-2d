try:
    import segmentation_models_pytorch as m
except ImportError:
    import models as m

import torch


def create_model(
        arch: str,
        encoder_name: str = 'resnet34',
        encoder_weights: str = 'imagenet',
        in_channels: int = 3,
        classes: int = 1,
        **kwargs
) -> torch.nn.Module:
    return m.create_model(
        arch,
        encoder_name,
        encoder_weights,
        in_channels,
        classes,
        **kwargs,
    )
