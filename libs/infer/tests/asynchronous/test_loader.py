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
    loader = FrameLoader(inference_sampling_rate, sample_rate, channels)
    yield loader
    loader.in_q.close()
    loader.out_q.close()
    loader.strain_q.close()


def test_frame_loader(
    write_dir, inference_sampling_rate, sample_rate, channels, loader
):
    num_frames = 10
    data = np.arange(num_frames * sample_rate * len(channels)).reshape(
        num_frames, len(channels), -1
    )

    for i, frame in enumerate(data):
        strain = TimeSeries(frame[0], channel=channels[0], dt=1 / sample_rate)
        witness = TimeSeriesDict(
            {i: j for i, j in zip(frame[1:], channels[1:])}
        )

        strain_fname = str(write_dir / f"strain-{i}.gwf")
        witness_fname = str(write_dir / f"witness-{i}.gwf")

        strain.write(strain_fname)
        witness.write(witness_fname)

        loader.in_q.put((witness_fname, strain_fname))

    expected_steps = int(num_frames * sample_rate // inference_sampling_rate)
    for i in range(expected_steps):
        try:
            package = loader.out_q.get_nowait()
        except Empty:
            raise ValueError(
                "Expected {} packages, only found {}".format(expected_steps, i)
            )

        assert package.request_id == i
        assert package.sequence_start == (i == 0)
        assert package.x.shape[0] == (len(channels) - 1)
        assert package.x.shape[-1] == int(
            sample_rate // inference_sampling_rate
        )

        for j in range(0, len(channels) - 1):
            k = sample_rate * (i * len(channels) + j + 1)
            assert (package.x[j] == np.arange(k, k + sample_rate)).all()

    with pytest.raises(Empty):
        loader.out_q.get_nowait()

    for i in range(num_frames):
        try:
            fnames, strain = loader.strain_q.get_nowait()
        except Empty:
            raise ValueError(
                "Expected {} strains, only found {}".format(num_frames, i)
            )
        witness_fname, strain_fname = fnames

        assert witness_fname.endswith(f"witness-{i}.gwf")
        assert strain_fname.endswith(f"strain-{i}.gwf")

        assert len(strain.shape) == 1
        assert len(strain) == sample_rate

        j = i * len(channels) * sample_rate
        assert (strain == np.arange(j, j + sample_rate)).all()
