from typing import Optional

import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: int,
        transpose: bool,
        output_padding: Optional[int] = None,
        activation: nn.Module = nn.Tanh
    ) -> None:
        super().__init__()

        if not transpose and output_padding is not None:
            raise ValueError(
                "Cannot specify output padding to non-transposed convolution"
            )
        elif output_padding is not None:
            kwargs = {"output_padding": output_padding}
        else:
            kwargs = {}

        conv_op = nn.ConvTranspose1d if transpose else nn.Conv1d
        self.conv = conv_op(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            **kwargs
        )
        self.bn = nn.BatchNorm1d(out_channels)
        self.activation = activation()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        return x

class DeepCleanAE(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        num_witnesses = in_channels
        self.input_conv = ConvBlock(
            in_channels,
            in_channels,
            kernel_size=7,
            stride=1,
            padding=3,
            transpose=False
        )

        self.downsampler = nn.Sequential()
        for i, out_channels in enumerate([8, 16, 32, 64]):
            conv_block = ConvBlock(
                in_channels,
                out_channels,
                kernel_size=7,
                stride=2,
                padding=3,
                transpose=False
            )
            self.downsampler.add_module(f"CONV_{i+1}", conv_block)
            in_channels = out_channels

        self.upsampler = nn.Sequential()
        for i, out_channels in enumerate([32, 16, 8, num_witnesses]):
            conv_block = ConvBlock(
                in_channels,
                out_channels,
                kernel_size=7,
                stride=2,
                padding=3,
                output_padding=1,
                transpose=True
            )
            self.upsampler.add_module(f"CONVTRANS_{i+1}", conv_block)
            in_channels = out_channels

        self.output_conv = nn.Conv1d(in_channels, 1, kernel_size=7, stride=1, padding=3)

    def forward(self, x):
        x = self.input_conv(x)
        x = self.downsampler(x)
        x = self.upsampler(x)
        x = self.output_conv(x)
        return x[:, 0]