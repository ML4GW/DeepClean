from unittest.mock import patch

import numpy as np
import pytest
import train
from gwpy.timeseries import TimeSeries, TimeSeriesDict


@pytest.fixture
def fake_freqs():
    return [18.37, 22.14]


@pytest.fixture
def fake_channels(fake_freqs):
    channels = []
    for freq in fake_freqs:
        freq = str(freq).replace(".", "POINT")
        channel = "FAKE_SINE_FREQ_" + freq + "HZ"
        channels.append(channel)
    return channels


@pytest.fixture
def fake_channel_test_fn(fake_freqs, fake_channels):
    def test_fn(result, t0, duration, sample_rate):
        time = np.arange(t0, t0 + duration, 1 / sample_rate)
        for freq, channel in zip(fake_freqs, fake_channels):
            y = result[channel]
            assert len(y) == (duration * sample_rate)

            expected = np.sin(2 * np.pi * freq * time)
            assert (expected == y).all()

    return test_fn


def test_make_fake_sines(fake_freqs, fake_channels, fake_channel_test_fn):
    t0 = 10
    duration = 100
    sample_rate = 256
    result = train.make_fake_sines(fake_channels, t0, duration, sample_rate)
    assert len(result) == 2
    fake_channel_test_fn(result, t0, duration, sample_rate)


def test_fetch(fake_channels, fake_channel_test_fn):
    t0 = 10
    duration = 100
    sample_rate = 256

    def get_patch(channels, t0, tf, **kwargs):
        x = np.arange(0, (tf - t0) * sample_rate, 0.5)

        result = {}
        for i, channel in enumerate(channels):
            result[channel] = TimeSeries(x + i, dt=0.5 / sample_rate)
        return TimeSeriesDict(result)

    real_channels = ["thom", "jonny"]
    channels = fake_channels + real_channels
    with patch("gwpy.timeseries.TimeSeriesDict.get", new=get_patch):
        result = train.fetch(channels, t0, duration, sample_rate)

    assert len(result) == 4
    fake_channel_test_fn(result, t0, duration, sample_rate)
    for i, channel in enumerate(real_channels):
        y = result[channel]
        assert len(y) == (duration * sample_rate)

        expected = np.arange(duration * sample_rate) + i
        assert np.isclose(y, expected, rtol=1e-9).all()


def test_read():
    return


def test_write():
    return


def test_main():
    return
