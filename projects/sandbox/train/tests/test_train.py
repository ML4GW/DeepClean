import logging
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
    logging.shutdown()
    shutil.rmtree(tmpdir)


@pytest.fixture
def t0():
    return 1234567890


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

    def test_fn(result, dur=duration, offset=0):
        time = np.arange(t0, t0 + dur, 1 / sample_rate) + offset
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
def real_channel_test_fn(real_channels, t0, duration, sample_rate, oversample):
    def test_fn(result, dur=duration, offset=0):
        for i, channel in enumerate(real_channels):
            y = result[channel]
            assert len(y) == (dur * sample_rate)

            expected = np.arange(dur * sample_rate) + i + offset * sample_rate
            if oversample == 1:
                assert np.isclose(y, expected, rtol=1e-9).all()
            else:
                assert np.isclose(y[4:-4:2], expected[4:-4:2], rtol=1e-9).all()

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
    dt = 1 / (sample_rate * oversample)
    ts_dict = {}
    for channel, waveform in zip(real_channels, real_waveforms):
        ts_dict[channel] = TimeSeries(waveform, dt=dt)
    ts_dict = TimeSeriesDict(ts_dict)

    # test the fetch function with the patched `get` method
    channels = fake_channels + real_channels
    with patch(
        "gwpy.timeseries.TimeSeriesDict.get", return_value=ts_dict
    ) as mock:
        result = train.fetch(channels, t0, duration, sample_rate)
    mock.assert_called_once_with(
        real_channels, t0, t0 + duration, nproc=4, allow_tape=True
    )

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
    train.write(data, fname, t0, sample_rate * oversample)

    with h5py.File(fname, "r") as f:
        assert len(list(f.keys())) == 2
        for channel, waveform in zip(real_channels, real_waveforms):
            dataset = f[channel]
            assert (dataset[:] == waveform).all()

            assert dataset.attrs["t0"] == t0
            assert dataset.attrs["sample_rate"] == sample_rate * oversample


@pytest.fixture(params=[None, 0.1, 0.25])
def valid_frac(request):
    return request.param


def test_main(
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
    valid_frac,
):
    channels = ["strain"] + real_channels + fake_channels
    strain = -np.arange(0, duration * sample_rate, 1 / oversample)

    def verify_strain(strain, witnesses, offset=0):
        assert strain.ndim == 1
        assert len(strain) == witnesses.shape[1]

        expected = -np.arange(len(strain)) - offset * sample_rate
        slc = slice(4, -4, 2) if oversample == 0.5 else slice(None, None)
        assert np.isclose(strain[slc], expected[slc], rtol=1e-9).all()

    def verify_outputs(outputs):
        if valid_frac is not None:
            assert len(outputs) == 4
            train_X, train_y, valid_X, valid_y = outputs

            train_dict = {
                channel: x for channel, x in zip(sorted(channels[1:]), train_X)
            }
            valid_dict = {
                channel: x for channel, x in zip(sorted(channels[1:]), valid_X)
            }

            train_length = (1 - valid_frac) * duration
            valid_length = duration - train_length
            real_channel_test_fn(train_dict, train_length)
            fake_channel_test_fn(train_dict, train_length)

            real_channel_test_fn(valid_dict, valid_length, train_length)
            fake_channel_test_fn(valid_dict, valid_length, train_length)

            # now check the strain data
            verify_strain(train_y, train_X)
            verify_strain(valid_y, valid_X, train_length)

            # TODO: check valid data too

        else:
            assert len(outputs) == 2
            train_X, train_y = outputs
            train_dict = {
                channel: x for channel, x in zip(sorted(channels[1:]), train_X)
            }
            real_channel_test_fn(train_dict)
            fake_channel_test_fn(train_dict)
            verify_strain(train_y, train_X)

    # create a patch for the TimeSeriesDict.get function
    # that just returns some pre-determined data
    dt = 1 / (sample_rate * oversample)
    ts_dict = {}
    for channel, waveform in zip(real_channels, real_waveforms):
        ts_dict[channel] = TimeSeries(waveform, dt=dt)
    ts_dict["strain"] = TimeSeries(strain, dt=dt)
    ts_dict = TimeSeriesDict(ts_dict)

    # create a dummy function which will generate the data
    # with `get` patched using different choices of `data_path`
    # and `force_download` and verify both the output as well
    # as whether `get` was called when it's expected to get called
    @patch("gwpy.timeseries.TimeSeriesDict.get", return_value=ts_dict)
    def run_fn(data_path, force_download, is_called, mock):
        output = train.main(
            channels,
            t0,
            duration,
            sample_rate,
            output_directory=tmpdir,
            data_path=data_path,
            valid_frac=valid_frac,
            force_download=force_download,
        )

        if is_called:
            mock.assert_called_once_with(
                ["strain"] + sorted(real_channels),
                t0,
                t0 + duration,
                nproc=4,
                allow_tape=True,
            )
        else:
            mock.assert_not_called()

        verify_outputs(output)

    # first case: data path is None
    run_fn(None, False, True)

    # second case: data path is a directory which exists
    # but which doesn't have the relevant file in it
    run_fn(tmpdir, False, True)

    # validate that the file got created
    fname = tmpdir / f"deepclean_train-{t0}-{duration}.h5"
    assert fname.is_file()

    # third case: data path is a directory which exists
    # and does contain the relevant file
    run_fn(tmpdir, False, False)

    # fourth case: data path is a directory which
    # exists and contains the relevant file,
    # but force_download is true
    run_fn(tmpdir, True, True)

    # fifth case: data path is an explicit filename
    # which already exists
    new_fname = tmpdir / "data.h5"
    shutil.move(fname, new_fname)
    run_fn(new_fname, False, False)

    # sixth case: data path is an explicit filename
    # which already exists but force_download=True
    run_fn(new_fname, True, True)

    # seventh case: data path is an explicit filename
    # which does _not_ exist, but which has a valid
    # filename format
    for suffix in ["h5", "hdf5"]:
        new_fname = tmpdir / f"some_prefix-{t0}-{duration}.{suffix}"

        # validate all of the potential exists/force_download
        # possibilities here
        run_fn(new_fname, False, True)
        assert new_fname.is_file()

        run_fn(new_fname, False, False)
        run_fn(new_fname, True, True)

        # make sure that nothing funky happens if force_download
        # is True but the file doesn't exist
        new_fname.unlink()
        run_fn(new_fname, True, True)
        assert new_fname.is_file()

    # eighth case: data path is an explicit filename
    # which does _not_ exist, but which has an _invalid_
    # filename format (ends in .gwf)
    bad_fname = tmpdir / f"some_prefix-{t0}-{duration}.gwf"
    with pytest.raises(ValueError) as exc_info:
        # is_called value shouldn't matter here because
        # a ValueError will be raised before we can check
        run_fn(bad_fname, False, None)
    assert str(exc_info.value) == f"Can't create data file {bad_fname}"

    # ninth and final case: data path is a directory
    # which does not exist yet, make sure it gets created
    # and contains the file
    subdir = tmpdir / "subdirectory"
    run_fn(subdir, False, True)

    assert subdir.is_dir()
    fname = subdir / f"deepclean_train-{t0}-{duration}.h5"
    assert fname.is_file()

    # double check that if the directory doesn't exist
    # but force_download is True, nothing funny happens
    fname.unlink()
    subdir.rmdir()
    run_fn(subdir, True, True)
    assert subdir.is_dir()
    assert fname.is_file()
