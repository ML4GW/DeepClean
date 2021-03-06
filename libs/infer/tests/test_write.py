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


@pytest.fixture(params=[50, 200])
def aggregation_steps(request):
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
def dataset(
    read_dir,
    write_dir,
    aggregation_steps,
    sample_rate,
    inference_sampling_rate,
    start_timestamp,
    num_frames,
    frame_length,
    channel_name,
):
    # build a dummy arange array for testing, but
    # add negative elements to the start that we
    # expect to get thrown away due to aggregation
    throw_away = int(aggregation_steps * sample_rate / inference_sampling_rate)
    x = np.arange(-throw_away, num_frames * frame_length * sample_rate)

    # break the non-negative parts of x into frame-sized
    # chunks that will be our dummy "strain"
    strains = np.split(x[throw_away:], num_frames)
    for i, strain in enumerate(strains):
        ts = TimeSeries(strain, dt=1 / sample_rate, channel=channel_name)
        tstamp = start_timestamp + i * frame_length
        fname = f"H1:STRAIN-{tstamp}-{frame_length}.gwf"
        ts.write(read_dir / fname)

    # break all of x up into update-sized chunks
    # that will be our dummy "witnesses," pass
    # all the updates as dummy package objects
    # into the writer's in_q for processing
    num_splits = (len(x) / sample_rate) * inference_sampling_rate
    if num_splits % 1 != 0:
        num_splits = int(num_splits)
        total_length = num_splits * int(sample_rate / inference_sampling_rate)
        x = x[:total_length]
    updates = np.split(x, num_splits)
    return updates


def test_writer(
    read_dir,
    write_dir,
    channel_name,
    inference_sampling_rate,
    sample_rate,
    postprocessor,
    memory,
    look_ahead,
    aggregation_steps,
    num_frames,
    frame_length,
    validate_fname,
    start_timestamp,
    dataset,
):
    writer = FrameWriter(
        read_dir,
        write_dir,
        channel_name=channel_name,
        inference_sampling_rate=inference_sampling_rate,
        sample_rate=sample_rate,
        postprocessor=postprocessor,
        memory=memory,
        look_ahead=look_ahead,
        aggregation_steps=aggregation_steps,
    )

    frame_idx = 0
    for i, update in enumerate(dataset):
        response = writer(update, i)
        if i < aggregation_steps:
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
    num_valid = len(dataset) - aggregation_steps
    size = num_valid * int(sample_rate / inference_sampling_rate)
    length, leftover = divmod(size, sample_rate)
    expected_frames = length // frame_length

    if leftover > (sample_rate * look_ahead):
        missing_frames = 0
    elif look_ahead < frame_length:
        missing_frames = 1
    else:
        missing_frames = 2

    expected_idx = expected_frames - missing_frames
    assert frame_idx == expected_idx, (size, i, leftover)
