from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
from gwpy.timeseries import TimeSeries

from deepclean.gwftools.frames import parse_frame_name
from deepclean.infer.write import FrameWriter


@pytest.fixture
def start_timestamp():
    return 1649289999


@pytest.fixture
def channel_name():
    return "test-channel"


@pytest.fixture(params=[0, 5, 20])
def aggregation_steps(request):
    return request.param


@pytest.fixture(params=[1, 3, 4])
def batch_size(request):
    return request.param


@pytest.fixture
def validate_fname(
    validate_frame, start_timestamp, frame_length, num_frames, channel_name
):
    def _validate_fname(fname, i):
        # make sure the frame name is correct
        _, t0, length = parse_frame_name(fname)
        assert t0 == (start_timestamp + i * frame_length)
        assert length == frame_length

        # the last frame will still get processed
        # because the internal writer._mask array
        # won't be long enough, but it will be wrong
        # because it won't include any "future" data.
        # This is just an artifact of the finite
        # processing scenario and won't come up in
        # production, so just exit the loop because
        # the validation will fail
        if (i + 1) == num_frames:
            return

        # now read in the written frame and validate
        # that it matches our expectations
        ts = TimeSeries.read(fname, channel=channel_name + "-CLEANED")
        assert ts.t0.value == t0
        assert (ts.dt.value * len(ts)) == length

        validate_frame(ts.value, i)
        return False

    return _validate_fname


@pytest.fixture
def get_strain_iter(start_timestamp):
    def f(sample_rate, num_frames, channel, frame_length=1):
        x = np.arange(num_frames * frame_length * sample_rate)

        def gen():
            strains = np.split(x, num_frames)
            for i, strain in enumerate(strains):
                tstamp = start_timestamp + i * frame_length
                fname = f"H1:STRAIN-{tstamp}-{frame_length}.gwf"
                yield strain, Path(fname)

        return x, iter(gen())

    return f


@pytest.fixture
def dataset(
    get_strain_iter,
    write_dir,
    aggregation_steps,
    sample_rate,
    inference_sampling_rate,
    start_timestamp,
    num_frames,
    frame_length,
    channel_name,
    batch_size,
):
    # build a dummy arange array for testing, but
    # add negative elements to the start that we
    # expect to get thrown away due to aggregation
    stride = int(sample_rate / inference_sampling_rate)
    throw_away = aggregation_steps * stride
    x, strain_iter = get_strain_iter(
        sample_rate, num_frames, channel_name, frame_length
    )
    prepend = np.arange(-throw_away, 0)
    x = np.concatenate([prepend, x])

    # break all of x up into update-sized chunks
    # that will be our dummy "witnesses," pass
    # all the updates as dummy package objects
    # into the writer's in_q for processing
    num_splits = len(x) / stride / batch_size
    if num_splits % 1 != 0:
        num_splits = int(num_splits)
        total_length = num_splits * batch_size * stride
        x = x[:total_length]
    updates = np.split(x, num_splits)

    return updates, strain_iter


def test_writer_validate_response(batch_size, aggregation_steps):
    writer = FrameWriter(
        MagicMock(),
        write_dir=MagicMock(),
        channel_name="",
        inference_sampling_rate=16,
        sample_rate=128,
        batch_size=batch_size,
        aggregation_steps=aggregation_steps,
    )

    x = np.random.randn(1, 8 * batch_size - 1)
    with pytest.raises(ValueError) as exc:
        writer.validate_response(x, 0)
    assert str(exc.value).startswith("Noise prediction is of wrong length")

    x = np.random.randn(1, 8 * batch_size)
    result = writer.validate_response(x, 0)
    if aggregation_steps == 0:
        assert result.shape == (x.shape[-1],)
    else:
        assert result is None

    result = writer.validate_response(x, 1)
    if aggregation_steps == 0:
        assert result.shape == (x.shape[-1],)
    elif aggregation_steps == 5:
        if batch_size == 1:
            assert result is None
        elif batch_size == 3:
            assert result.shape == (8,)
        else:
            assert result.shape == (24,)
    elif aggregation_steps == 20:
        assert result is None
    else:
        raise ValueError(
            "aggregation_steps value is {}, "
            "update testing code".format(aggregation_steps)
        )

    result = writer.validate_response(x, 2)
    if aggregation_steps == 0:
        assert result.shape == (x.shape[-1],)
    elif aggregation_steps < 20 and batch_size > 1:
        assert result.shape == (x.shape[-1],)
    else:
        assert result is None


def test_writer_update_prediction_array(batch_size, aggregation_steps):
    writer = FrameWriter(
        MagicMock(),
        write_dir=MagicMock(),
        channel_name="",
        inference_sampling_rate=16,
        sample_rate=128,
        batch_size=batch_size,
        aggregation_steps=aggregation_steps,
    )

    x = np.random.randn(1, 8 * batch_size)
    response = writer.validate_response(x, 0)
    if aggregation_steps == 0:
        idx = writer.update_prediction_array(response, 0)
        assert idx == batch_size
        assert len(writer._noise) == 128
        assert (writer._noise[: 8 * batch_size] == x[0]).all()

    response = writer.validate_response(x, 1)
    if aggregation_steps == 20:
        assert response is None
    elif batch_size > 1 or aggregation_steps == 0:
        idx = writer.update_prediction_array(response, 1)

        if aggregation_steps == 0:
            expected = np.concatenate([x[0], x[0]])
            assert idx == (2 * batch_size), idx
            assert (writer._noise[: 16 * batch_size] == expected).all()
        elif batch_size == 3:
            assert idx == 1, idx
            assert (writer._noise[:8] == x[0, -8:]).all()
            assert (writer._noise[8:] == 0).all()
        elif batch_size == 4:
            assert idx == 3, idx
            assert (writer._noise[:24] == x[0, -24:]).all()
            assert (writer._noise[24:] == 0).all()

    if not (aggregation_steps == 20 and batch_size == 4):
        return

    for i in range(2, 5):
        response = writer.validate_response(x, i)
        assert response is None, i

    response = writer.validate_response(x, 5)
    idx = writer.update_prediction_array(response, 5)
    assert idx == batch_size, (idx, batch_size)
    assert (writer._noise[: 8 * batch_size] == x[0]).all()
    assert (writer._noise[8 * batch_size :] == 0).all()


def test_writer_call(
    write_dir,
    channel_name,
    inference_sampling_rate,
    sample_rate,
    postprocessor,
    memory,
    look_ahead,
    aggregation_steps,
    batch_size,
    num_frames,
    frame_length,
    validate_fname,
    start_timestamp,
    dataset,
):
    x, strain_iter = dataset
    writer = FrameWriter(
        strain_iter,
        write_dir,
        channel_name=channel_name,
        inference_sampling_rate=inference_sampling_rate,
        sample_rate=sample_rate,
        batch_size=batch_size,
        postprocessor=postprocessor,
        memory=memory,
        look_ahead=look_ahead,
        aggregation_steps=aggregation_steps,
    )

    frame_idx = 0
    for i, update in enumerate(x):
        response = writer(update, i)
        if (i + 1) * batch_size < aggregation_steps:
            assert len(writer._noise) == 0
            continue

        if response is not None:
            fname, _ = response
            validate_fname(fname, frame_idx)
            frame_idx += 1

    # some convoluted logic here to deal with the case
    # that inference sampling rate is not an even factor
    # of the sample rate, which we don't actually support
    # right now. The issue I belive is happening in the
    # aggregation steps that are getting ditched, since
    # the case of look_ahead == frame_length is causing
    # problems in this context
    num_valid = len(x) * batch_size - aggregation_steps
    size = num_valid * int(sample_rate / inference_sampling_rate)
    expected_frames, leftover = divmod(size, sample_rate * frame_length)

    if leftover > (sample_rate * look_ahead):
        # we have enough future data leftover to be able
        # to process the last of the frames we expected
        missing_frames = 0
    elif look_ahead <= frame_length:
        # we didn't have enough future data, but the current
        # frame provided enough future data to finish processing
        # the previous one
        missing_frames = 1
    else:
        # trying to look ahead too far, not enought data
        # to process either of the last two frames
        missing_frames = 2

    expected_idx = expected_frames - missing_frames
    assert frame_idx == expected_idx
