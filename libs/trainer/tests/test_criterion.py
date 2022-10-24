import numpy as np
import pytest
import scipy
import torch
from packaging import version
from scipy import signal

from deepclean.trainer.criterion import PSDLoss, TorchWelch


@pytest.fixture(params=[1, 4, 8])
def length(request):
    return request.param


@pytest.fixture(params=[1024, 4096])
def sample_rate(request):
    return request.param


@pytest.mark.parametrize(
    "overlap,fftlength",
    [[None, 2], [0.5, 2], [2, 2], [None, 4], [2, 4], [4, 4]],
)
def test_welch_init(overlap, fftlength, sample_rate):
    if overlap is not None and overlap >= fftlength:
        with pytest.raises(ValueError):
            welch = TorchWelch(sample_rate, fftlength, overlap)
        return
    else:
        welch = TorchWelch(sample_rate, fftlength, overlap)

    if overlap is None:
        expected_stride = int(fftlength * sample_rate // 2)
    else:
        expected_stride = int(fftlength * sample_rate) - int(
            overlap * sample_rate
        )
    assert welch.nstride == expected_stride


@pytest.fixture(params=[0.5, 2, 4])
def fftlength(request):
    return request.param


@pytest.fixture(params=[None, 0.1, 0.5, 1])
def overlap(request):
    return request.param


@pytest.fixture(params=[None, 55, [25, 55]])
def freq_low(request):
    return request.param


@pytest.fixture(params=[None, 65, [35, 65]])
def freq_high(request):
    return request.param


@pytest.fixture(params=[True, False])
def fast(request):
    return request.param


@pytest.fixture(params=["mean", "median"])
def average(request):
    return request.param


@pytest.fixture(params=[1, 2, 3])
def ndim(request):
    return request.param


def test_welch(length, sample_rate, fftlength, overlap, fast, average, ndim):
    batch_size = 8
    num_channels = 5
    if overlap is not None and overlap >= fftlength:
        return

    welch = TorchWelch(
        sample_rate, fftlength, overlap, average=average, fast=fast
    )

    shape = [int(length * sample_rate)]
    if ndim > 1:
        shape.insert(0, num_channels)
    if ndim > 2:
        shape.insert(0, batch_size)
    x = np.random.randn(*shape)

    # make sure we catch if the fftlength is too long for the data
    if fftlength > length:
        with pytest.raises(ValueError) as exc_info:
            welch(torch.Tensor(x))
        assert str(exc_info.value).startswith("Number of samples")
        return

    # perform the transform and confirm the shape is correct
    torch_result = welch(torch.Tensor(x)).numpy()
    num_freq_bins = int(fftlength * sample_rate) // 2 + 1
    shape[-1] = num_freq_bins
    assert torch_result.shape == tuple(shape)

    # now verify against the result from scipy
    _, scipy_result = signal.welch(
        x,
        fs=sample_rate,
        nperseg=welch.nperseg,
        noverlap=welch.nperseg - welch.nstride,
        window=signal.windows.hann(welch.nperseg, False),
        average=average,
    )

    # if we're using the fast implementation, only guarantee
    # that components higher than the first two are correct
    if fast:
        torch_result = torch_result[..., 2:]
        scipy_result = scipy_result[..., 2:]
    assert np.isclose(torch_result, scipy_result, rtol=1e-9).all()

    # make sure we catch any calls with too many dimensions
    if ndim == 3:
        with pytest.raises(ValueError) as exc_info:
            welch(torch.Tensor(x[None]))
        assert str(exc_info.value).startswith("Can't perform welch")


@pytest.fixture(params=[0, 1])
def y_ndim(request):
    return request.param


def _shape_checks(ndim, y_ndim, x, y, welch):
    # verify that time dimensions must match
    with pytest.raises(ValueError) as exc_info:
        welch(x, y[..., :-1])
    assert str(exc_info.value).startswith("Time dimensions")

    # verify that y can't have more dims than x
    if y_ndim == 0:
        with pytest.raises(ValueError) as exc_info:
            welch(x, y[None])
        assert str(exc_info.value).startswith("Can't compute")

        if ndim == 1:
            assert "1D" in str(exc_info.value)

    # verify that if x is greater than 1D and y has
    # the same dimensionality, their shapes must
    # fully match
    if ndim > 1 and y_ndim == 0:
        with pytest.raises(ValueError) as exc_info:
            welch(x, y[:-1])
        assert str(exc_info.value).startswith("If x and y tensors")

    # verify for 3D x's that 2D y's must have the same batch
    # dimension, and that y cannot be 1D
    if ndim == 3 and y_ndim == 1:
        with pytest.raises(ValueError) as exc_info:
            welch(x, y[:-1])
        assert str(exc_info.value).startswith("If x is a 3D tensor")

        with pytest.raises(ValueError) as exc_info:
            welch(x, y[0])
        assert str(exc_info.value).startswith("Can't compute cross")


def test_welch_with_csd(
    y_ndim, length, sample_rate, fftlength, overlap, average, ndim, fast
):
    batch_size = 8
    num_channels = 5
    if overlap is not None and overlap >= fftlength:
        return

    if y_ndim == 1 and ndim == 1:
        return

    welch = TorchWelch(
        sample_rate, fftlength, overlap, average=average, fast=fast
    )

    shape = [int(length * sample_rate)]
    if ndim > 1:
        shape.insert(0, num_channels)
    if ndim > 2:
        shape.insert(0, batch_size)
    x = np.random.randn(*shape)

    if ndim == 1 or (y_ndim == 1 and ndim == 2):
        y = np.random.randn(shape[-1])
    elif ndim == 3 and y_ndim == 1:
        y = np.random.randn(shape[0], shape[-1])
    else:
        y = np.random.randn(*shape)

    x = torch.Tensor(x)
    y = torch.Tensor(y)

    # make sure we catch if the fftlength is too long for the data
    if fftlength > length:
        with pytest.raises(ValueError) as exc_info:
            welch(x, y)
        assert str(exc_info.value).startswith("Number of samples")
        return
    elif not fast:
        with pytest.raises(NotImplementedError):
            welch(x, y)
        return

    # perform the transform and confirm the shape is correct
    torch_result = welch(x, y).numpy()
    num_freq_bins = int(fftlength * sample_rate) // 2 + 1
    shape[-1] = num_freq_bins
    assert torch_result.shape == tuple(shape)

    if ndim == 3:
        scipy_result = []
        if y_ndim == 1:
            x = x.transpose(1, 0)
            y = [y] * len(x)

        for i, j in zip(x, y):
            _, result = signal.csd(
                i,
                j,
                fs=sample_rate,
                nperseg=welch.nperseg,
                noverlap=welch.nperseg - welch.nstride,
                window=signal.windows.hann(welch.nperseg, False),
                average=average,
            )
            scipy_result.append(result)
        scipy_result = np.stack(scipy_result)

        if y_ndim == 1:
            x = x.transpose(1, 0)
            y = y[0]
            scipy_result = scipy_result.transpose(1, 0, 2)
    else:
        _, scipy_result = signal.csd(
            x,
            y,
            fs=sample_rate,
            nperseg=welch.nperseg,
            noverlap=welch.nperseg - welch.nstride,
            window=signal.windows.hann(welch.nperseg, False),
            average=average,
        )
    assert scipy_result.shape == torch_result.shape

    scipy_version = version.parse(scipy.__version__)
    num_windows = (x.shape[-1] - welch.nperseg) // welch.nstride + 1
    if (
        average == "median"
        and scipy_version < version.parse("1.9")
        and num_windows > 1
    ):
        # scipy actually had a bug in the median calc for
        # csd, see this issue:
        # https://github.com/scipy/scipy/issues/15601
        from scipy.signal.spectral import _median_bias

        scipy_result *= _median_bias(num_freq_bins)
        scipy_result /= _median_bias(num_windows)

    if fast:
        torch_result = torch_result[..., 2:]
        scipy_result = scipy_result[..., 2:]

    ratio = torch_result / scipy_result
    assert np.isclose(torch_result, scipy_result, rtol=1e-9).all(), ratio

    _shape_checks(ndim, y_ndim, x, y, welch)


def test_psd_loss(
    length, sample_rate, fftlength, overlap, freq_low, freq_high
):
    # check all the conditions that would cause
    # a problem at instantiation time
    will_raise = False
    will_raise |= overlap is not None and overlap >= fftlength
    will_raise |= freq_low is None and freq_high is not None
    will_raise |= freq_low is not None and freq_high is None
    will_raise |= isinstance(freq_low, list) and not isinstance(
        freq_high, list
    )
    will_raise |= isinstance(freq_high, list) and not isinstance(
        freq_low, list
    )

    if will_raise:
        with pytest.raises(ValueError):
            criterion = PSDLoss(
                sample_rate,
                fftlength,
                overlap,
                freq_low=freq_low,
                freq_high=freq_high,
            )
        return
    else:
        criterion = PSDLoss(
            sample_rate,
            fftlength,
            overlap,
            freq_low=freq_low,
            freq_high=freq_high,
        )

    if freq_low is None:
        assert criterion.mask is None
        return

    try:
        num_ranges = len(freq_low)
    except TypeError:
        num_ranges = 1
        freq_low = [freq_low]
        freq_high = [freq_high]

    freqs_per_bin = criterion.welch.nperseg / sample_rate
    in_range_bins = int(10 * num_ranges * freqs_per_bin)
    assert criterion.mask.sum() == in_range_bins

    # make sure that any bins marked as valid in
    # the mask correspond to at least one of the
    # desired frequency ranges, allowing some slack
    nz_idx = np.where(criterion.mask > 0)[0]
    nz_freqs = nz_idx / freqs_per_bin
    alright = np.zeros_like(nz_idx, dtype=bool)
    for low, high in zip(freq_low, freq_high):
        # allow a little slack at the edges to account
        # for discretization noise
        in_range = 0.99 * low < nz_freqs
        in_range &= 1.01 * high > nz_freqs
        alright |= in_range
    assert alright.all()

    # now test to make sure that the criterion evaluates
    # to roughly what we would expect. Do this by bandstop
    # filtering a time series in the range we're evaluating
    # in. This should mean the time series is roughly 0 in
    # those bins, and therefore the ratio should be rougly 1.
    # We can handle multiple ranges in a sort of gross manner
    # by first bandstop filtering from the lowest low to the
    # highest high, then adding in a bandpass filter in
    # the middle areas
    x = np.random.randn(8, int(length * sample_rate))

    # start by bandstop filtering the widest range possible
    low = freq_low[0]
    high = freq_high[-1]
    sos = signal.butter(
        32, [low, high], btype="bandstop", output="sos", fs=sample_rate
    )
    y = signal.sosfiltfilt(sos, x)

    # if we have more than one range, bandpass in the
    # middle region and average this filtered time
    # series with the bandstopped one
    if len(freq_low) > 1:
        low = freq_high[0]
        high = freq_low[1]
        sos = signal.butter(
            32, [low, high], btype="bandpass", output="sos", fs=sample_rate
        )
        y += signal.sosfiltfilt(sos, x)
        y /= 2

    # move these timeseries into torch
    x = torch.Tensor(x)
    y = torch.Tensor(y.copy())

    # check for runtime errors
    if fftlength > length:
        with pytest.raises(ValueError):
            criterion(y, x)
        return

    # evaluate the psd loss using these timeseries.
    # since `y` should be roughly 0. in the relevant
    # frequency bins, `(x - y) / x` should be ~1 everywhere
    result = criterion(y, x).numpy()
    assert 0.75 < result < 1.25
