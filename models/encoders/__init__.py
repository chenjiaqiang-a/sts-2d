import functools
import torch.utils.model_zoo as model_zoo

from .resnet import resnet_encoders
from .efficientnet import efficient_net_encoders
from .senet import senet_encoders
from .densenet import densenet_encoders

from ._preprocessing import preprocess_input

encoders = {}
encoders.update(resnet_encoders)
encoders.update(efficient_net_encoders)
encoders.update(senet_encoders)
encoders.update(densenet_encoders)


def get_encoder(name, in_channels=3, depth=5, weights=None, output_stride=32):

    try:
        Encoder = encoders[name]["encoder"]
    except KeyError:
        raise KeyError("Wrong encoder name `{}`, supported encoders: {}".format(name, list(encoders.keys())))

    params = encoders[name]["params"]
    params.update(depth=depth)
    encoder = Encoder(**params)

    if weights is not None:
        try:
            settings = encoders[name]["pretrained_settings"][weights]
        except KeyError:
            raise KeyError(
                "Wrong pretrained weights `{}` for encoder `{}`. Available options are: {}".format(
                    weights,
                    name,
                    list(encoders[name]["pretrained_settings"].keys()),
                )
            )
        encoder.load_state_dict(model_zoo.load_url(settings["url"]))

    encoder.set_in_channels(in_channels, pretrained=weights is not None)
    if output_stride != 32:
        encoder.make_dilated(output_stride)

    return encoder


def get_encoder_names():
    return list(encoders.keys())


def get_preprocessing_params(encoder_name, pretrained="imagenet"):

    all_settings = encoders[encoder_name]["pretrained_settings"]
    if pretrained not in all_settings.keys():
        raise ValueError("Available pretrained options {}".format(all_settings.keys()))
    settings = all_settings[pretrained]

    formatted_settings = {"input_space": settings.get("input_space", "RGB"),
                          "input_range": list(settings.get("input_range", [0, 1])), "mean": list(settings["mean"]),
                          "std": list(settings["std"])}

    return formatted_settings


def get_preprocessing_fn(encoder_name, pretrained="imagenet"):
    params = get_preprocessing_params(encoder_name, pretrained=pretrained)
    return functools.partial(preprocess_input, **params)