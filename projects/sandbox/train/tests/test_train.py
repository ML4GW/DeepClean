import shutil
from pathlib import Path
from unittest.mock import patch

import h5py
import numpy as np
import pytest
import train
from gwpy.timeseries import TimeSeries, TimeSeriesDict


@pytest.fixture(scope="function")
def tmpdir():
    tmpdir = Path(__file__).resolve().parent / "tmp"
    tmpdir.mkdir(parents=True, exist_ok=False)
    yield tmpdir
    shutil.rmtree(tmpdir)


@pytest.fixture
def t0():
    return 10


@pytest.fixture
def duration():
    return 100


@pytest.fixture
def sample_rate():
    return 256


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
def fake_channel_test_fn(fake_freqs, fake_channels, t0, duration, sample_rate):
    """Create a fixture function for validating fake sinusoids"""

    time = np.arange(t0, t0 + duration, 1 / sample_rate)

    def test_fn(result, dur=duration):
        for freq, channel in zip(fake_freqs, fake_channels):
            y = result[channel]
            assert len(y) == (dur * sample_rate)

            expected = np.sin(2 * np.pi * freq * time)
            assert (expected == y).all()

    return test_fn


@pytest.fixture
def real_channels():
    return ["thom", "jonny"]


@pytest.fixture(params=[1, 2, 0.5])
def oversample(request):
    return request.param


@pytest.fixture
def real_waveforms(real_channels, t0, duration, sample_rate, oversample):
    x = np.arange(0, duration * sample_rate, 1 / oversample)
    return [x + i for i in range(len(real_channels))]


@pytest.fixture
def real_channel_test_fn(real_channels, t0, duration, sample_rate):
    def test_fn(result, dur=duration):
        for i, channel in enumerate(real_channels):
            y = result[channel]
            assert len(y) == (dur * sample_rate)

            expected = np.arange(dur * sample_rate) + i
            assert np.isclose(y, expected, rtol=1e-9).all()

    return test_fn


def test_make_fake_sines(
    fake_freqs, fake_channels, fake_channel_test_fn, t0, duration, sample_rate
):
    result = train.make_fake_sines(fake_channels, t0, duration, sample_rate)
    assert len(result) == 2
    fake_channel_test_fn(result)


def test_fetch(
    real_channels,
    real_waveforms,
    real_channel_test_fn,
    fake_channels,
    fake_channel_test_fn,
    t0,
    duration,
    sample_rate,
    oversample,
):
    # create a patch for the TimeSeriesDict.get function
    # that just returns some pre-determined data
    def get_patch(channels, t0, tf, **kwargs):
        result = {}
        for channel, waveform in zip(real_channels, real_waveforms):
            result[channel] = TimeSeries(
                waveform, dt=(sample_rate * oversample) ** -1
            )
        return TimeSeriesDict(result)

    # test the fetch function with the patched `get` method
    channels = fake_channels + real_channels
    with patch("gwpy.timeseries.TimeSeriesDict.get", new=get_patch) as mock:
        result = train.fetch(channels, t0, duration, sample_rate)
    mock.assert_called_once_with(real_channels, t0, t0 + duration)

    # make sure we have all the channels we expect
    assert len(result) == len(channels)

    # verify that the data for each set of channels
    # got created correctly
    fake_channel_test_fn(result)
    real_channel_test_fn(result)


def test_read(
    tmpdir,
    real_channels,
    real_waveforms,
    real_channel_test_fn,
    fake_channels,
    fake_channel_test_fn,
    t0,
    duration,
    sample_rate,
    oversample,
):
    fname = tmpdir / "data.h5"
    with h5py.File(fname, "w") as f:
        for channel, waveform in zip(real_channels, real_waveforms):
            dataset = f.create_dataset(channel, data=waveform)
            dataset.attrs["sample_rate"] = sample_rate * oversample
            dataset.attrs["t0"] = t0

    channels = real_channels + fake_channels
    result = train.read(fname, channels, t0, duration, sample_rate)
    assert len(result) == len(channels)

    fake_channel_test_fn(result)
    real_channel_test_fn(result)

    # ensure that using a shorter duration is allowed
    result = train.read(fname, channels, t0, duration - 1, sample_rate)
    assert len(result) == len(channels)

    fake_channel_test_fn(result, duration - 1)
    real_channel_test_fn(result, duration - 1)

    # ensure that using too _long_ of a duration raises an error
    with pytest.raises(ValueError) as exc_info:
        train.read(fname, channels, t0, duration + 1, sample_rate)
    assert f"{duration}s worth of data" in str(exc_info)

    # make sure that non-matching t0s raises an error
    with pytest.raises(ValueError) as exc_info:
        train.read(fname, channels, t0 - 1, duration, sample_rate)
    assert f"has initial GPS timestamp {t0}" in str(exc_info)


def test_write(
    tmpdir,
    real_channels,
    real_waveforms,
    fake_channels,
    t0,
    sample_rate,
    oversample,
):
    data = {channel: None for channel in fake_channels}
    data.update({i: j for i, j in zip(real_channels, real_waveforms)})

    fname = tmpdir / "data.h5"
    train.write(data, fname, sample_rate * oversample, t0)

    with h5py.File(fname, "r") as f:
        assert len(list(f.keys())) == 2
        for channel, waveform in zip(real_channels, real_waveforms):
            dataset = f[channel]
            assert (dataset[:] == waveform).all()

            assert dataset.attrs["t0"] == t0
            assert dataset.attrs["sample_rate"] == sample_rate * oversample


def test_main():
    return
