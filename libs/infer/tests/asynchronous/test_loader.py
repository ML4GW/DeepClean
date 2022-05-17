import logging
import time
from queue import Empty

import numpy as np
import pytest
from gwpy.timeseries import TimeSeries, TimeSeriesDict

from deepclean.infer.asynchronous import FrameLoader


@pytest.fixture(params=[128, 768, 1024])
def inference_sampling_rate(request):
    return request.param


@pytest.fixture(params=[512, 4096])
def sample_rate(request):
    return request.param


@pytest.fixture(params=["A", ["A"], list("ABCDEFG")])
def channels(request):
    if isinstance(request.param, str):
        return "channel" + request.param
    return [f"channel{i}" for i in request.param]


@pytest.fixture
def loader(inference_sampling_rate, sample_rate, channels):
    loader = FrameLoader(
        inference_sampling_rate, sample_rate, channels, name="loader"
    )
    loader.logger = logging.getLogger()
    yield loader
    loader.in_q.close()
    loader.out_q.close()


@pytest.fixture
def ts(sample_rate):
    def _ts(x, channel):
        return TimeSeries(x, channel=channel, dt=1 / sample_rate)

    return _ts


def test_frame_loader(
    write_dir, inference_sampling_rate, sample_rate, channels, loader, ts
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
    # and put the filenames into the loader's in_q
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
        loader.in_q.put(fname)

    # put in a StopIteration to keep loader.get_package()
    # from waiting for a new package indefinitely
    loader.in_q.put(StopIteration)

    stride = int(sample_rate // inference_sampling_rate)
    num_updates = int(num_frames * sample_rate // stride)
    for i in range(num_updates):
        # simulate the stillwater run loop, making sure
        # to catch the stop iteration on the last element
        package = loader.get_package()
        if (i + 1) == num_updates:
            with pytest.raises(StopIteration):
                loader.process(package)
        else:
            loader.process(package)
        time.sleep(0.01)

        # get the package from the loader's out_q
        try:
            package = loader.out_q.get_nowait()
        except Empty:
            raise ValueError(
                f"Expected {num_updates} packages, only found {i}"
            )

        # make sure it matches all our expectations
        assert package.request_id == i
        assert package.sequence_start == (i == 0)
        assert package.sequence_end == ((i + 1) == num_updates)
        if not isinstance(channels, str):
            assert package.x.shape[0] == num_channels
        assert package.x.shape[-1] == stride

        if isinstance(channels, str):
            # verify shape and add dimension to make
            # loop below more general
            assert len(package.x.shape) == 1
            package.x = package.x[None]

        # make sure the channel content is correct
        for j in range(0, num_channels):
            k = j * num_frames * sample_rate
            expected = np.arange(k + i * stride, k + (i + 1) * stride)
            assert (package.x[j] == expected).all()

    # make sure there's nothing else in the out_q
    with pytest.raises(Empty):
        loader.out_q.get_nowait()
