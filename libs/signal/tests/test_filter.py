import numpy as np
import pytest

from deepclean.signal.filter import BandpassFilter, normalize_frequencies


@pytest.mark.parametrize("iterable_type", [list, tuple])
def test_normalize_frequencies(iterable_type):
    # ensure basic behavior maps to lists
    assert normalize_frequencies(55, 65) == ([55], [65])

    # ensure iterables map to lists
    assert normalize_frequencies(
        iterable_type([55, 75]), iterable_type([65, 85])
    ) == ([55, 75], [65, 85])

    # ensure lists are organized
    assert normalize_frequencies(
        iterable_type([75, 55]), iterable_type([85, 65])
    ) == ([55, 75], [65, 85])

    # float w/ iterable raises error
    with pytest.raises(ValueError):
        normalize_frequencies(55, iterable_type([65]))
    with pytest.raises(ValueError):
        normalize_frequencies(iterable_type([55]), 65)

    # low > high raises error
    with pytest.raises(ValueError):
        normalize_frequencies(65, 55)

    # mismatched lengths raises error
    with pytest.raises(ValueError):
        normalize_frequencies(iterable_type([55, 75]), iterable_type([65]))

    # iterable with one low > high raises error
    with pytest.raises(ValueError):
        normalize_frequencies(iterable_type([65, 75]), iterable_type([55, 85]))

    # iterable with overlapping bands raises error
    with pytest.raises(ValueError):
        normalize_frequencies(iterable_type([55, 60]), iterable_type([65, 70]))


def validate_at_precision(y, expected, precision):
    err = np.abs(y - expected) / np.abs(expected)
    assert np.percentile(err, 99) < precision


@pytest.mark.parametrize("sample_rate", [512, 4096])
def test_bandpass_filter(sample_rate):
    # set up a signal with two frequency components
    x = np.linspace(0, 16, sample_rate * 16)
    y_60 = np.sin(60 * 2 * np.pi * x)
    y = y_60 + np.sin(80 * 2 * np.pi * x)

    # only look at the middle of the signal to
    # ignore filtering edge effects
    slc = slice(sample_rate * 2, sample_rate * 14)

    # bandpass one of the frequency components and
    # ensure that roughly only one is left
    bandpass = BandpassFilter(55, 65, sample_rate)
    y_hat = bandpass(y)
    validate_at_precision(y_hat[slc], y_60[slc], 1e-5)

    # keep both components and ensure that
    # they're both still present
    bandpass = BandpassFilter([55, 75], [65, 85], sample_rate)
    y_hat = bandpass(y)
    validate_at_precision(y_hat[slc], y[slc], 1e-5)
