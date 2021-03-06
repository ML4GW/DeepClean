import shutil
from pathlib import Path

import pytest


@pytest.fixture(scope="function")
def tmpdir():
    tmpdir = Path(__file__).resolve().parent / "tmp"
    tmpdir.mkdir(parents=True)
    yield tmpdir
    try:
        shutil.rmtree(tmpdir)
    except Exception:
        raise ValueError(list(tmpdir.iterdir()))


@pytest.fixture(params=[True])  # , False])
def write_dir_exists(request):
    return request.param


@pytest.fixture(scope="function")
def write_dir(tmpdir, write_dir_exists):
    write_dir = tmpdir / "write"
    if write_dir_exists:
        write_dir.mkdir(parents=True)
    return write_dir


@pytest.fixture(scope="function")
def read_dir(tmpdir):
    read_dir = tmpdir / "read"
    read_dir.mkdir(parents=True)
    return read_dir


@pytest.fixture(params=[10])  # , 100])
def num_frames(request):
    return request.param


@pytest.fixture(params=[1, 4])
def frame_length(request):
    return request.param


@pytest.fixture(params=[1024, 4096])
def sample_rate(request):
    return request.param


@pytest.fixture(params=[128, 1024])
def inference_sampling_rate(request):
    # TODO: support non-factors of sample_rate
    return request.param


@pytest.fixture(params=[2, 10])
def memory(request):
    return request.param


@pytest.fixture(params=[0.1, 1])
def look_ahead(request):
    return request.param


@pytest.fixture
def postprocessor():
    """
    postprocessor just subtracts the mean, and frame
    gets subtracted from itself. So we should expect
    every sample in the "cleaned" output to be equal
    to the mean across the postprocessing window
    """

    def _postprocessor(x, **kwargs):
        return x - x.mean()

    return _postprocessor


@pytest.fixture
def validate_frame(frame_length, sample_rate, memory, look_ahead):
    # convert times to sizes
    frame_size = int(frame_length * sample_rate)
    filter_lead_size = int(look_ahead * sample_rate)
    filter_memory_size = int(memory * sample_rate)

    def _validate_frame(frame, i):
        # make sure frame has appropriate length
        assert len(frame) == (frame_length * sample_rate)

        # N is the last sample of the postprocessing window
        # n is first sample of the filter memory
        N = (i + 1) * frame_size + filter_lead_size
        n = max(i * frame_size - filter_memory_size, 0)
        expected = (N + n - 1) / 2

        # ensure that all samples in the frame
        # equal this expected value
        assert (frame == expected).all()

    return _validate_frame
