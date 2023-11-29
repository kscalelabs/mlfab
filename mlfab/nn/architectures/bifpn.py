"""Defines a BiFPN model architecture.

The BiFPN model takes in a list of feature maps from a backbone and outputs a
list of feature maps that have been fused together using a bidirectional
feature pyramid network. This is a good general-purpose model for multi-scale
per-pixel classification or prediction.

.. code-block:: python

    bsz = image.shape[0]
    x1, x2, x3, x4 = resnet_backbone(image)
    assert x1.shape[1] == 256
    assert x2.shape[1] == 512
    assert x3.shape[1] == 1024
    assert x4.shape[1] == 2048
    bifpn = BiFPN([256, 512, 1024, 2048], feature_size=32)
    f1, f2, f3, f4 = bifpn([x1, x2, x3, x4])
    assert f1.shape == (bsz, 32, *x1.shape[2:])
    assert f2.shape == (bsz, 32, *x2.shape[2:])
    assert f3.shape == (bsz, 32, *x3.shape[2:])
    assert f4.shape == (bsz, 32, *x4.shape[2:])
"""

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class DepthwiseConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 1,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
    ) -> None:
        super().__init__()

        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=in_channels,
            bias=False,
        )

        self.pointwise = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            bias=False,
        )

        self.bn = nn.BatchNorm2d(out_channels, momentum=0.9997, eps=4e-5)
        self.act = nn.ReLU()

    def forward(self, x: Tensor) -> Tensor:
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 1,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
    ) -> None:
        super().__init__()

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )
        self.bn = nn.BatchNorm2d(out_channels, momentum=0.9997, eps=4e-5)
        self.act = nn.ReLU()

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class BiFPNBlock(nn.Module):
    __constants__ = ["scale_factor"]

    def __init__(
        self,
        feature_size: int,
        num_inputs: int,
        scale_factor: float,
    ) -> None:
        super().__init__()

        self.scale_factor = scale_factor

        self.in_convs = nn.ModuleList([ConvBlock(feature_size, feature_size) for _ in range(num_inputs - 1)])
        self.out_convs = nn.ModuleList([ConvBlock(feature_size, feature_size) for _ in range(num_inputs - 1)])
        self.w1 = nn.Parameter(torch.ones(2, num_inputs - 1))
        self.w2 = nn.Parameter(torch.ones(3, num_inputs - 1))

    def forward(self, inputs: list[Tensor]) -> list[Tensor]:
        num_inputs = len(inputs)
        assert len(self.in_convs) == num_inputs - 1
        assert len(self.out_convs) == num_inputs - 1

        # Validates the input shapes.
        for i in range(num_inputs - 1):
            assert inputs[i].shape[2] == int(inputs[i + 1].shape[2] * self.scale_factor)
            assert inputs[i].shape[3] == int(inputs[i + 1].shape[3] * self.scale_factor)

        # Computes BiFPN layer weights.
        w1 = self.w1.softmax(dim=1)
        w2 = self.w2.softmax(dim=1)

        # Computes downward path.
        feature = inputs[0]
        features = [feature]
        for i in range(num_inputs - 1):
            features_downsampled = F.interpolate(feature, scale_factor=1 / self.scale_factor)
            feature = self.in_convs[i](w1[0, i] * inputs[i + 1] + w1[1, i] * features_downsampled)
            features.append(feature)

        # Computes upward path.
        out_features = [feature]
        for i in range(num_inputs - 2, -1, -1):
            features_upsampled = F.interpolate(feature, scale_factor=self.scale_factor)
            feature = self.out_convs[i](w2[0, i] * inputs[i] + w2[1, i] * features[i] + w2[2, i] * features_upsampled)
            out_features.append(feature)

        return out_features[::-1]


class BiFPN(nn.Module):
    """Defines the BiFPN module.

    This implementation assumes a constant downsampling factor of 2, and that
    the inputs are sorted in order from least to most downsampled, meaning that
    the first input should have the largest height and width.

    The inputs to the BiFPN are typically the outputs of some feature extractor
    backbone like ResNet, and the outputs of the BiFPN are typically used as
    inputs to some head like a classification head. This can be used for
    multi-scale per-pixel classification or prediction.

    Parameters:
        input_size: The number of channels for each input.
        feature_size: The number of channels for the BiFPN features. All of the
            BiFPN output features will have this number of channels.
        num_layers: The number of layers to use for each BiFPN block.
        scale_factor: The downsampling factor between input images. This is
            assumed to be the same for all inputs.

    Inputs:
        inputs: A list of tensors with shape ``(batch_size, channels, height,
            width)``, where the tensors are sorted in order from least to most
            downsampled.
    """

    def __init__(
        self,
        input_size: list[int],
        feature_size: int = 64,
        num_layers: int = 2,
        scale_factor: float = 2.0,
    ) -> None:
        super().__init__()

        self.convs = nn.ModuleList([DepthwiseConvBlock(s, feature_size) for s in input_size])
        bifpns = []
        for _ in range(num_layers):
            bifpns.append(BiFPNBlock(feature_size, len(input_size), scale_factor))
        self.bifpn = nn.Sequential(*bifpns)

    def forward(self, inputs: list[Tensor]) -> list[Tensor]:
        assert len(inputs) == len(self.convs)
        features = [conv(x) for conv, x in zip(self.convs, inputs)]
        return self.bifpn(features)
