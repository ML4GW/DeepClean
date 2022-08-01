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
