import numpy as np
import pytest
import torch

from deepclean.export import PrePostDeepClean


def test_pre_post_deepclean():
    deepclean = torch.nn.Sequential(
        torch.nn.Conv1d(4, 2, 4), torch.nn.ConvTranspose1d(2, 1, 4)
    )

    # models without a num_witnesses attribute
    # will raise a ValueError
    with pytest.raises(ValueError):
        prepost = PrePostDeepClean(deepclean)

    deepclean.num_witnesses = 4
    prepost = PrePostDeepClean(deepclean)

    # make sure that parameters are included in the state dict
    assert len(list(deepclean.state_dict().keys())) == 4
    assert len(list(prepost.state_dict().keys())) == 8

    # make sure that before fitting the
    # prepost model's behavior is the same
    # as standalone deepclean
    x = torch.randn(8, 4, 100)
    assert (deepclean(x) == prepost(x)).numpy().all()

    X = np.stack([np.arange(100) + i for i in range(6)])
    y = np.arange(100)

    # X has too many channels right now
    with pytest.raises(ValueError):
        prepost.fit(X, y)

    X = X[:4]
    prepost.fit(X, y)

    # make sure that the parameters didn't get
    # changed to vanilla tensors during fitting
    assert len(list(prepost.state_dict().keys())) == 8

    # make sure the fit parameter values are correct
    # TODO: should we do something more explicit here?
    assert np.isclose(
        prepost.input_shift.numpy()[:, 0], X.mean(axis=1), rtol=1e-6
    ).all()
    assert np.isclose(
        prepost.input_scale.numpy()[:, 0], X.std(axis=1), rtol=1e-6
    ).all()
    assert prepost.output_shift.numpy() == y.mean()
    assert prepost.output_scale.numpy() == y.std()

    # now run the pre/post steps manually for comparison
    expected = (x - prepost.input_shift) / prepost.input_scale
    expected = deepclean(expected)
    expected = prepost.output_scale * expected + prepost.output_shift

    # ensure that the preprost model performs all steps
    with torch.no_grad():
        output = prepost(x).numpy()
        assert np.isclose(output, expected.numpy(), rtol=1e-6).all()
