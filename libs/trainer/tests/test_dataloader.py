import numpy as np
import pytest

from deepclean.trainer.dataloader import ChunkedTimeSeriesDataset


@pytest.fixture(params=[50])
def length(request):
    return request.param


@pytest.fixture(params=[1024, 4096])
def sample_rate(request):
    return request.param


@pytest.fixture(params=[1, 2, 8])
def kernel_length(request):
    return request.param


@pytest.fixture(params=[0.05, 0.1, 0.25])
def kernel_stride(request):
    return request.param


@pytest.fixture(params=[1, 8, 32, 128])
def batch_size(request):
    return request.param


@pytest.fixture(params=[-1, 0, 0.15, 5])
def chunk_length(request):
    return request.param


@pytest.fixture(params=[1, 5])
def num_chunks(request):
    return request.param


def test_dataloader(
    length,
    sample_rate,
    kernel_length,
    kernel_stride,
    batch_size,
    chunk_length,
    num_chunks,
):
    if chunk_length <= 0 and num_chunks != 1:
        return

    size = int(length * sample_rate)
    kernel_size = int(kernel_length * sample_rate)
    stride_size = int(kernel_stride * sample_rate)

    X = np.random.randn(21, size)
    y = np.random.randn(size)

    # if we're planning on chunking, make sure that the
    # appropriate errors around the relative lengths
    # of chunks and kernels are raised
    if chunk_length > 0:
        # convert a 0 < chunk_length < 1 chunk_length to
        # its "actual" value to see if an error will get
        # raised
        if chunk_length >= 1:
            effective_chunk_length = chunk_length
        else:
            effective_chunk_length = chunk_length * length

        # how many kernels will appear in a combined chunk
        kernels_per_chunk = (
            effective_chunk_length * num_chunks - kernel_length
        ) // kernel_stride + 1

        # initialization will raise a ValueError if the kernel
        # length is longer than the length of a chunk, or if
        # there will not be enough kernels in a combined
        # chunk to be able to generate a full batch of data
        will_raise = kernel_length >= effective_chunk_length
        will_raise |= kernels_per_chunk < batch_size
        if will_raise:
            with pytest.raises(ValueError):
                dataloader = ChunkedTimeSeriesDataset(
                    X,
                    y,
                    kernel_length=kernel_length,
                    kernel_stride=kernel_stride,
                    sample_rate=sample_rate,
                    batch_size=batch_size,
                    chunk_length=chunk_length,
                    num_chunks=num_chunks,
                    shuffle=False,
                )
            return

    # don't shuffle so that we know what to expect
    # each kernel to be. TODO: how to test shuffle?
    dataloader = ChunkedTimeSeriesDataset(
        X,
        y,
        kernel_length=kernel_length,
        kernel_stride=kernel_stride,
        sample_rate=sample_rate,
        batch_size=batch_size,
        chunk_length=chunk_length,
        num_chunks=num_chunks,
        shuffle=False,
    )

    samples_seen = 0
    chunk_offset = 0
    idx_in_chunk = 0
    chunk_idx = 0

    for i, (_x, _y) in enumerate(dataloader):
        # make sure the batch dimension is equal to the
        # batch size for every batch except the last
        if (i + 1) < len(dataloader):
            try:
                assert _x.shape[0] == batch_size
                assert _y.shape[0] == batch_size
            except AssertionError:
                print(i + 1, len(dataloader))
        else:
            # make sure this last batch is at most batch_size
            assert _x.shape[0] <= batch_size
            assert _y.shape[0] <= batch_size

        # now check remaining dimensions
        assert _x.shape[1:] == (21, kernel_size)
        assert _y.shape[1:] == (kernel_size,)

        # keep track of the number of samples we've observed
        samples_seen += len(_x)

        # now verify that the actual tensors themselves all match up
        _x = _x.numpy()
        _y = _y.numpy()
        for j in range(len(_x)):
            # keep track of the start idx separately
            # when we're chunking since that will skip
            # over possible frames at the edges between
            # chunks
            if chunk_length <= 0:
                start = (i * batch_size + j) * stride_size
            else:
                start = idx_in_chunk * stride_size + chunk_offset
            stop = start + kernel_size

            assert np.isclose(_x[j], X[:, start:stop], rtol=1e-6).all()
            assert np.isclose(_y[j], y[start:stop], rtol=1e-6).all()

            # for chunking, keep track of when we move
            # from one chunk to the next, since this will
            # skip over kernels that might otherwise
            # be sample-able
            if chunk_length > 0:
                idx_in_chunk += 1
                if idx_in_chunk == dataloader.get_num_kernels(
                    dataloader.X[chunk_idx].shape[-1]
                ):
                    idx_in_chunk = 0
                    chunk_offset += dataloader.X[chunk_idx].shape[-1]
                    chunk_idx += 1

    assert samples_seen == dataloader.num_kernels
