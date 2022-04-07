import numpy as np
import pytest

from deepclean.signal.filter import BandpassFilter, normalize_frequencies


def test_normalize_frequencies():
    assert normalize_frequencies(55, 65) == ([55], [65])
    assert normalize_frequencies([55, 75], [65, 85]) == ([55, 75], [65, 85])
    assert normalize_frequencies([75, 55], [85, 65]) == ([55, 75], [65, 85])

    with pytest.raises(ValueError):
        normalize_frequencies(55, [65])
    with pytest.raises(ValueError):
        normalize_frequencies([55], 65)
    with pytest.raises(ValueError):
        normalize_frequencies(65, 55)
    with pytest.raises(ValueError):
        normalize_frequencies([55, 75], [65])
    with pytest.raises(ValueError):
        normalize_frequencies([65, 75], [55, 85])
    with pytest.raises(ValueError):
        normalize_frequencies([55, 60], [65, 70])


@pytest.fixture(params=[512, 4096])
def sample_rate(request):
    return request.param


def validate_at_precision(y, expected, precision):
    err = np.abs(y - expected) / np.abs(expected)
    assert np.percentile(err, 99) < precision


def test_bandpass_filter(sample_rate):
    x = np.linspace(0, 16, sample_rate * 16)
    y_60 = np.sin(60 * 2 * np.pi * x)
    y = y_60 + np.sin(80 * 2 * np.pi * x)

    bandpass = BandpassFilter(55, 65, sample_rate)
    y_hat = bandpass(y)

    slc = slice(sample_rate * 2, sample_rate * 14)
    validate_at_precision(y_hat[slc], y_60[slc], 1e-5)

    bandpass = BandpassFilter([55, 75], [65, 85], sample_rate)
    y_hat = bandpass(y)
    validate_at_precision(y_hat[slc], y[slc], 1e-5)
