import numpy as np
import torch

from deepclean.export import PrePostDeepClean


def test_pre_post_deepclean():
    deepclean = torch.nn.Sequential(
        torch.nn.Conv1d(4, 2, 4), torch.nn.ConvTranspose1d(2, 4, 4)
    )
    nn = PrePostDeepClean(deepclean)
    assert len(list(deepclean.state_dict().keys())) == 4
    assert len(list(nn.state_dict().keys())) == 8

    x = torch.randn(8, 4, 100)
    assert (deepclean(x) == nn(x)).numpy().all()

    X = np.stack([np.arange(100) + i for i in range(4)])
    y = np.arange(100)
    nn.fit(X, y)

    assert len(list(nn.state_dict().keys())) == 8
    assert np.isclose(
        nn.input_shift.numpy()[:, 0], X.mean(axis=1), rtol=1e-6
    ).all()
    assert np.isclose(
        nn.input_scale.numpy()[:, 0], X.std(axis=1), rtol=1e-6
    ).all()
    assert nn.output_shift.numpy() == y.mean()
    assert nn.output_scale.numpy() == y.std()

    expected = (x - nn.input_shift) / nn.input_scale
    expected = deepclean(expected)
    expected = nn.output_scale * expected + nn.output_shift

    with torch.no_grad():
        output = nn(x).numpy()
        assert np.isclose(output, expected.numpy(), rtol=1e-6).all()
