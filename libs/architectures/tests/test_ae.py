import pytest
import torch

from deepclean.architectures import DeepCleanAE


@pytest.fixture(params=[1, 10, 21])
def in_channels(request):
    return request.param


@pytest.fixture(params=[128, 512, 2048, 8192])
def length(request):
    return request.param


def test_ae(in_channels, length):
    deepclean = DeepCleanAE(in_channels)
    x = torch.randn(8, in_channels, length)
    y = deepclean(x)
    assert y.shape == (8, length)
