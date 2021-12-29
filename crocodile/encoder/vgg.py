from typing import Union, List, Dict, Any, cast

import torch
import torch.nn as nn
from .encoder import Encoder
from dataclasses import dataclass
from enum import Enum
from simple_parsing.helpers import Serializable
from simple_parsing.helpers.serialization import encode, register_decoding_fn


class VGGType(Enum):
    VGG11 = "vgg11"
    VGG13 = "vgg13"
    VGG16 = "vgg16"
    VGG19 = "vgg19"


@encode.register
def encode_vgg_type(obj: VGGType) -> str:
    """ We choose to encode a tensor as a list, for instance """
    return obj.name


def decode_vgg_type(name: str) -> VGGType:
    return VGGType[name]


register_decoding_fn(VGGType, decode_vgg_type)


@dataclass
class VGGOptions(Serializable):
    vgg: VGGType = VGGType.VGG11
    batchnorm: bool = True

class VGG(Encoder):
    def __init__(self, options: VGGOptions = VGGOptions()):
        super().__init__()
        self.options = options

    def build(self, input_dim: int, output_dim: int):
        self.network = _vgg(self.options.vgg, self.options.batchnorm, num_class=output_dim)

    def forward(self, x):
        return self.network(x)


class _VGG(nn.Module):
    def __init__(
        self, features: nn.Module, num_classes: int = 1000, init_weights: bool = True, dropout: float = 0.5
    ) -> None:
        super().__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool1d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, num_classes),
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def make_layers(cfg: List[Union[str, int]], batch_norm: bool = False) -> nn.Sequential:
    layers: List[nn.Module] = []
    in_channels = 3
    for v in cfg:
        if v == "M":
            layers += [nn.MaxPool1d(kernel_size=2, stride=2)]
        else:
            v = cast(int, v)
            conv1d = nn.Conv1d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv1d, nn.BatchNorm1d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv1d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfgs: Dict[VGGType, List[Union[str, int]]] = {
    VGGType.VGG11: [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    VGGType.VGG13: [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    VGGType.VGG16: [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"],
    VGGType.VGG19: [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512, "M"],
}


def _vgg(type: VGGType, batch_norm: bool, pretrained: bool, **kwargs: Any) -> VGG:
    if pretrained:
        kwargs["init_weights"] = False
    model = VGG(make_layers(cfgs[type], batch_norm=batch_norm), **kwargs)
    return model