import numpy as np
import pytest
from gwpy.timeseries import TimeSeries, TimeSeriesDict

from deepclean.infer import load


@pytest.fixture
def ts(sample_rate):
    def _ts(x, channel):
        return TimeSeries(x, channel=channel, dt=1 / sample_rate)

    return _ts


@pytest.fixture(params=["A", ["A"], list("ABCDEFG")])
def channels(request):
    if isinstance(request.param, str):
        return "channel" + request.param
    return [f"channel{i}" for i in request.param]


def test_load_frame(write_dir, channels, ts, sample_rate):
    x = np.arange(sample_rate * 4)
    if isinstance(channels, str):
        data = ts(x, channels)
    else:
        data = [ts(x + i, channel) for i, channel in enumerate(channels)]
        data = TimeSeriesDict({i: j for i, j in zip(channels, data)})

    fname = write_dir / "data.gwf"
    data.write(fname)

    # test read with same sample rate
    result = load.load_frame(fname, channels, sample_rate)
    if isinstance(channels, str):
        assert result.shape == (len(x),)
        assert (result == x).all()
    else:
        assert result.shape == (len(channels), len(x))
        for i, row in enumerate(result):
            assert (row == (x + i)).all()

    # now test read with lower sample rate
    result = load.load_frame(fname, channels, sample_rate / 2)
    if isinstance(channels, str):
        assert result.shape == (len(x) / 2,)
        assert (result[5:-5] == x[::2][5:-5]).all()
    else:
        assert result.shape == (len(channels), len(x) / 2)
        for i, row in enumerate(result):
            assert (row[5:-5] == (x[::2][5:-5] + i)).all()


def test_frame_iterator(
    write_dir, channels, ts, sample_rate, inference_sampling_rate
):
    # TODO: turn this into a check in loader __init__
    if sample_rate < inference_sampling_rate:
        return

    if isinstance(channels, str):
        num_channels = 1
    else:
        num_channels = len(channels)

    # create a neat organized dummy array to check against
    num_frames = 10
    data = (
        np.arange(num_frames * sample_rate * num_channels)
        .reshape(num_channels, num_frames, -1)
        .transpose(1, 0, 2)
    )
    if isinstance(channels, str):
        data = data[:, 0]

    # write frames using the dummy data
    fnames = []
    for i, frame in enumerate(data):
        if isinstance(channels, str):
            timeseries = ts(frame, channels)
        else:
            assert len(frame.shape) == 2
            timeseries = TimeSeriesDict(
                {i: ts(j, i) for i, j in zip(channels, frame)}
            )

        fname = write_dir / f"{i}.gwf"
        timeseries.write(fname)
        fnames.append(fname)

    crawler = fnames
    it = load.frame_iterator(
        crawler, channels, sample_rate, inference_sampling_rate
    )
    stride = int(sample_rate // inference_sampling_rate)
    num_updates = int(num_frames * sample_rate // stride)

    for i, (x, sequence_end) in enumerate(it):
        if isinstance(channels, str):
            assert x.shape == (stride,)
            assert (x == np.arange(stride) * (i + 1)).all()
        else:
            assert x.shape == (num_channels, stride)
            for j, row in enumerate(x):
                start = i * num_channels + j
                stop = start + stride
                assert (x == np.arange(start, stop)).all()

        assert sequence_end == ((i + 1) == num_updates)
    assert sequence_end
