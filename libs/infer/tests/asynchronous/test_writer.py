import logging
from dataclasses import dataclass
from queue import Empty, Queue

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


@pytest.fixture(params=[128, 512, 1024])
def inference_sampling_rate(request):
    return request.param


@pytest.fixture(params=[50, 200])
def aggregation_steps(request):
    return request.param


@pytest.fixture
def writer(
    write_dir,
    inference_sampling_rate,
    sample_rate,
    memory,
    look_ahead,
    aggregation_steps,
    postprocessor,
):
    writer = FrameWriter(
        write_dir,
        channel_name="test_channel",
        inference_sampling_rate=inference_sampling_rate,
        sample_rate=sample_rate,
        strain_q=Queue(),
        postprocessor=postprocessor,
        memory=memory,
        look_ahead=look_ahead,
        aggregation_steps=aggregation_steps,
        output_name="aggregator",
        name="writer",
    )
    writer.logger = logging.getLogger()

    yield writer
    writer.in_q.close()
    writer.out_q.close()


def test_writer(
    num_frames,
    frame_length,
    sample_rate,
    look_ahead,
    validate_frame,
    start_timestamp,
    write_dir,
    inference_sampling_rate,
    aggregation_steps,
    writer,
):
    # build a dummy arange array for testing, but
    # add negative elements to the start that we
    # expect to get thrown away due to aggregation
    throw_away = int(aggregation_steps * sample_rate / inference_sampling_rate)
    x = np.arange(-throw_away, num_frames * frame_length * sample_rate)

    # break the non-negative parts of x into frame-sized
    # chunks that will be our dummy "strain"
    strains = np.split(x[throw_away:], num_frames)

    # break all of x up into update-sized chunks
    # that will be our dummy "witnesses"
    updates = np.split(x, (len(x) / sample_rate) * inference_sampling_rate)

    # pass the strains and dummy filenames
    # into the writer's strain_q
    for i, strain in enumerate(strains):
        fname = f"{start_timestamp + i}-{frame_length}.gwf"
        witness_fname = write_dir / ("witness_" + fname)
        strain_fname = write_dir / ("strain_" + fname)

        # the package the writer expects in the strain
        # queue is a tuple of a (tuple of matching witness
        # and strain filenames) and the strain frame array
        fnames = (str(witness_fname), str(strain_fname))
        package = (fnames, strain)
        writer.strain_q.put(package)

        # need to be able to os.stat witness fnames for latency
        with open(witness_fname, "w"):
            pass

    # now pass all the updates as dummy package objects
    # into the writer's main in_q for processing
    for i, update in enumerate(updates):
        package = Package(update, i)
        writer.in_q.put({writer.output_name: package})

    # add in a `None` for our purposes here
    # so that we know when we're done
    writer.in_q.put(None)

    # now simulate the main `run` loop of
    # a stillwater process
    while True:
        package = writer.get_package()
        if package is None:
            break
        writer.process(package)

    # now ensure that the writer wrote each one of
    # the corresponding frames in order and run
    # our standard frame validater to ensure that
    # the frames have the right shape and content
    for i in range(num_frames):
        try:
            fname, latency = writer.out_q.get_nowait()
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

        # make sure the frame name is correct
        _, t0, length = parse_frame_name(fname)
        assert t0 == (start_timestamp + i)
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
            break

        # now read in the written frame and validate
        # that it matches our expectations
        ts = TimeSeries.read(fname, channel=writer.channel_name)
        assert ts.t0.value == t0
        assert (ts.dt.value * len(ts)) == length

        validate_frame(ts.value, i)

    # ensure that the queue is empty
    with pytest.raises(Empty):
        writer.out_q.get_nowait()
