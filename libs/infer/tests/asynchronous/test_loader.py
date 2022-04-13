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


@pytest.fixture
def channels():
    return [f"channel{i}" for i in "ABCDEFG"]


@pytest.fixture
def loader(inference_sampling_rate, sample_rate, channels):
    loader = FrameLoader(
        inference_sampling_rate, sample_rate, channels, name="loader"
    )
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

    # create a neat organized dummy array to check against
    num_frames = 10
    data = (
        np.arange(num_frames * sample_rate * len(channels))
        .reshape(len(channels), num_frames, -1)
        .transpose(1, 0, 2)
    )

    # write witness and strain frames using the dummy data
    # and put the filenames into the loader's in_q
    for i, frame in enumerate(data):
        strain = ts(frame[0], channels[0])
        witness = TimeSeriesDict(
            {i: ts(j, i) for i, j in zip(channels[1:], frame[1:])}
        )

        strain_fname = str(write_dir / f"strain-{i}.gwf")
        witness_fname = str(write_dir / f"witness-{i}.gwf")

        strain.write(strain_fname)
        witness.write(witness_fname)

        loader.in_q.put((witness_fname, strain_fname))

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
        assert package.x.shape[0] == (len(channels) - 1)
        assert package.x.shape[-1] == stride

        # make sure the channel content is correct
        for j in range(0, len(channels) - 1):
            # offset for the (j+1)th channel
            k = (j + 1) * num_frames * sample_rate
            expected = np.arange(k + i * stride, k + (i + 1) * stride)
            assert (package.x[j] == expected).all()

    # make sure there's nothing else in the out_q
    with pytest.raises(Empty):
        loader.out_q.get_nowait()

    # now process the data the loader put into its
    # strain q during that last loop
    for i in range(num_frames):
        try:
            fnames, strain = loader.strain_q.get_nowait()
        except Empty:
            raise ValueError(
                "Expected {} strains, only found {}".format(num_frames, i)
            )
        witness_fname, strain_fname = fnames

        # make sure the filenames are properly ordered and named
        assert witness_fname.endswith(f"witness-{i}.gwf")
        assert strain_fname.endswith(f"strain-{i}.gwf")

        # make sure the strain has the right shape and content
        assert len(strain.shape) == 1
        assert len(strain) == sample_rate

        expected = np.arange(i * sample_rate, (i + 1) * sample_rate)
        assert (strain == expected).all()
