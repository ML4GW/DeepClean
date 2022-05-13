import logging
from dataclasses import dataclass
from queue import Empty

import numpy as np
import pytest
from gwpy.timeseries import TimeSeries

from deepclean.gwftools.frames import parse_frame_name
from deepclean.infer.asynchronous import FrameWriter


@dataclass(frozen=True)
class Package:
    x: np.ndarray
    request_id: int


@pytest.fixture
def start_timestamp():
    return 1649289999


@pytest.fixture
def channel_name():
    return "test-channel"


@pytest.fixture(params=[128, 512, 1024])
def inference_sampling_rate(request):
    return request.param


@pytest.fixture(params=[50, 200])
def aggregation_steps(request):
    return request.param


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
    updates = np.split(x, (len(x) / sample_rate) * inference_sampling_rate)
    packages = []
    for i, update in enumerate(updates):
        package = Package(update, i)
        packages.append(package)
    return packages


def pass_dataset(writer, dataset):
    for update in dataset:
        writer.in_q.put({writer.output_name: update})


@pytest.fixture(scope="function")
def writer(
    read_dir,
    write_dir,
    channel_name,
    inference_sampling_rate,
    sample_rate,
    memory,
    look_ahead,
    aggregation_steps,
    postprocessor,
    dataset,  # noqa
):
    return FrameWriter(
        read_dir,
        write_dir,
        channel_name=channel_name,
        inference_sampling_rate=inference_sampling_rate,
        sample_rate=sample_rate,
        postprocessor=postprocessor,
        memory=memory,
        look_ahead=look_ahead,
        aggregation_steps=aggregation_steps,
        output_name="aggregator",
        name="writer",
    )


@pytest.fixture
def sync_writer(writer, dataset):
    writer.logger = logging.getLogger()
    try:
        pass_dataset(writer, dataset)
        yield writer
    finally:
        for q in [writer.in_q, writer.out_q]:
            while True:
                try:
                    q.get_nowait()
                except Empty:
                    break
            q.close()
            q.join_thread()


@pytest.fixture
def async_writer(writer, dataset):
    with writer:
        pass_dataset(writer, dataset)
        yield writer


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
            return True

        # now read in the written frame and validate
        # that it matches our expectations
        ts = TimeSeries.read(fname, channel=channel_name + "-CLEANED")
        assert ts.t0.value == t0
        assert (ts.dt.value * len(ts)) == length

        validate_frame(ts.value, i)
        return False

    return _validate_fname


def test_writer(
    num_frames,
    frame_length,
    look_ahead,
    validate_fname,
    start_timestamp,
    sync_writer,
    dataset,
):
    writer = sync_writer

    # add in a `None` for our purposes here
    # so that we know when we're done
    writer.in_q.put(None)

    # now simulate the main `run` loop of
    # a stillwater process
    for package in iter(writer.get_package, None):
        writer.process(package)

    # now ensure that the writer wrote each one of
    # the corresponding frames in order and run
    # our standard frame validater to ensure that
    # the frames have the right shape and content
    for i in range(num_frames - 1):
        try:
            fname, _ = writer.out_q.get_nowait()
        except Empty:
            # if the filter lead time is >= the
            # frame length, the (num_frames - 1)th frame
            # won't have gotten processed because
            # there wasn't sufficient data
            if look_ahead >= frame_length:
                break

            raise ValueError(
                "Expected {} frames but only found {}".format(num_frames, i)
            )
        validate_fname(fname, i)

    # ensure that the queue is empty
    with pytest.raises(Empty):
        writer.out_q.get_nowait()


def test_writer_async(
    num_frames,
    frame_length,
    look_ahead,
    validate_fname,
    start_timestamp,
    async_writer,
    dataset,
):
    writer = async_writer
    for i, (fname, _) in enumerate(writer):
        if validate_fname(fname, i):
            break
        elif (i + 2) == num_frames and look_ahead >= frame_length:
            break
