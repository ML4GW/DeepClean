import numpy as np
import pytest
import torch
from scipy import signal

from deepclean.trainer.criterion import PSDLoss, TorchWelch


@pytest.fixture(params=[1, 4, 8])
def length(request):
    return request.param


@pytest.fixture(params=[1024, 4096])
def sample_rate(request):
    return request.param


@pytest.fixture(params=[0.5, 2, 4])
def fftlength(request):
    return request.param


@pytest.fixture(params=[0.1, 0.5, 1])
def overlap(request):
    return request.param


@pytest.fixture(params=[None, 55, [25, 55]])
def freq_low(request):
    return request.param


@pytest.fixture(params=[None, 65, [35, 65]])
def freq_high(request):
    return request.param


def test_welch(length, sample_rate, fftlength, overlap):
    batch_size = 8
    min_freq_match = 5

    if overlap >= fftlength:
        with pytest.raises(ValueError):
            torch_welch = TorchWelch(sample_rate, fftlength, overlap)
        return
    else:
        torch_welch = TorchWelch(sample_rate, fftlength, overlap)

    x = np.random.randn(batch_size, int(length * sample_rate))

    if fftlength > length:
        with pytest.raises(ValueError):
            torch_welch(torch.Tensor(x))
        return
    else:
        torch_result = torch_welch(torch.Tensor(x)).numpy()

    _, scipy_result = signal.welch(
        x,
        fs=sample_rate,
        nperseg=torch_welch.nperseg,
        noverlap=torch_welch.noverlap,
        window=signal.windows.hann(torch_welch.nperseg, False)
    )
    ratio = torch_result / scipy_result

    # just enforce that all measurements greater than
    # some minimum frequency bin are very nearly constant,
    # since our loss function just takes the ratio
    freq_bins = np.linspace(0, sample_rate / 2, torch_welch.nfreq)
    ratio = ratio[:, freq_bins >= min_freq_match]
    ratio = ratio - ratio.mean()
    assert ratio.std() < 1e-6


def test_psd_loss(
    length,
    sample_rate,
    fftlength,
    overlap,
    freq_low,
    freq_high
):
    # check all the conditions that would cause
    # a problem at instantiation time
    will_raise = False
    will_raise |= overlap >= fftlength
    will_raise |= freq_low is None and freq_high is not None
    will_raise |= freq_low is not None and freq_high is None
    will_raise |= isinstance(freq_low, list) and not isinstance(freq_high, list)
    will_raise |= isinstance(freq_high, list) and not isinstance(freq_low, list)

    if will_raise:
        with pytest.raises(ValueError):
            criterion = PSDLoss(
                sample_rate,
                fftlength,
                overlap,
                freq_low=freq_low,
                freq_high=freq_high
            )
        return
    else:
        criterion = PSDLoss(
            sample_rate,
            fftlength,
            overlap,
            freq_low=freq_low,
            freq_high=freq_high
        )

    if freq_low is None:
        assert criterion.mask is None
    else:
        try:
            num_ranges = len(freq_low)
        except TypeError:
            num_ranges = 1
            freq_low = [freq_low]
            freq_high = [freq_high]

        freqs_per_bin = sample_rate / (2 * criterion.welch.nfreq)
        in_range_bins = int(10 * num_ranges / freqs_per_bin)
        assert criterion.N == in_range_bins

        # make sure that any bins marked as valid in
        # the mask correspond to at least one of the
        # desired frequency ranges, allowing some slack
        nz_idx = np.where(criterion.mask > 0)[0] 
        nz_freqs = nz_idx * freqs_per_bin
        alright = np.zeros_like(nz_idx, dtype=bool)
        for low, high in zip(freq_low, freq_high):
            # allow a little slack at the edges to account
            # for discretization noise
            in_range = (0.99 * low < nz_freqs)
            in_range &= (1.01 * high > nz_freqs)
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
        assert np.isclose(result, 1., rtol=0.1)
