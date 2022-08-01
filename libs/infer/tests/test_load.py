import numpy as np
import pytest
from gwpy.timeseries import TimeSeriesDict

from hermes.infer import load


@pytest.fixture(params=["A", ["A"], list("ABCDEFG")])
def channels(request):
    if isinstance(request.param, str):
        return "channel" + request.param
    return [f"channel{i}" for i in request.param]


def test_load_frame(write_dir, channels, ts, sample_rate):
    data = TimeSeriesDict(
        {
            channel: np.arange(sample_rate * 4) + i
            for i, channel in enumerate(channels)
        }
    )
    data.write(write_dir / "data.gwf")

    result = load(write_dir / "data.gwf", channels, sample_rate)
    if isinstance(channels, str):
        assert result.shape == (sample_rate * 4,)
        assert (result == np.arange(sample_rate * 4)).all()
    else:
        assert result.shape == (len(channels), sample_rate * 4)
        for i, row in enumerate(result):
            assert (row == np.arange(sample_rate * 4) + i).all()

    result = load(write_dir / "data.gwf", channels, sample_rate / 2)
    if isinstance(channels, str):
        assert result.shape == (sample_rate * 2,)
        assert (result == np.arange(0, sample_rate * 4, 2)).all()
    else:
        assert result.shape == (len(channels), sample_rate * 2)
        for i, row in enumerate(result):
            assert (row == np.arange(0, sample_rate * 4, 2) + i).all()
