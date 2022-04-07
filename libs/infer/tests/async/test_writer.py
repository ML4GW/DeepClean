import os
import pickle
import shutil
from pathlib import Path
from queue import Empty, Queue
from unittest.mock import Mock

import numpy as np
import pytest
from gwpy.timeseries import TimeSeries

from deepclean.infer.asynchronous import FrameWriter


@pytest.fixture
def start_timestamp():
    return 1649289999


@pytest.fixture
def write_dir():
    tmp_dir = Path(__file__).resolve().parent / "tmp"
    os.makedirs(tmp_dir)
    yield tmp_dir
    shutil.rmtree(tmp_dir)


@pytest.fixture(params=[128, 512, 1024])
def inference_sampling_rate(request):
    return request.param


@pytest.fixture(params=[50, 200])
def aggregation_steps(request):
    return request.param


def postprocessor(x, **kwargs):
    return x - x.mean()


def test_writer(
    num_frames,
    frame_length,
    sample_rate,
    filter_lead_time,
    filter_memory,
    validate_frame,
    start_timestamp,
    write_dir,
    inference_sampling_rate,
    aggregation_steps,
    # postprocessor
):
    postprocess_pkl = write_dir / "postproc.pkl"
    with open(postprocess_pkl, "wb") as f:
        pickle.dump(postprocessor, f)

    channel_name = "test_channel"
    strain_q = Queue()
    output_name = "aggregator"

    writer = FrameWriter(
        write_dir,
        channel_name=channel_name,
        inference_sampling_rate=inference_sampling_rate,
        sample_rate=sample_rate,
        strain_q=strain_q,
        postprocess_pkl=postprocess_pkl,
        memory=filter_memory,
        look_ahead=filter_lead_time,
        aggregation_steps=aggregation_steps,
        output_name=output_name,
        name="writer",
    )

    throw_away = int(aggregation_steps * sample_rate / inference_sampling_rate)
    x = np.arange(-throw_away, num_frames * frame_length * sample_rate)

    strains = np.split(x[throw_away:], num_frames)
    updates = np.split(x, (len(x) / sample_rate) * inference_sampling_rate)

    for i, strain in enumerate(strains):
        fname = f"{start_timestamp + i}_1.gwf"
        witness_fname = write_dir / ("witness-" + fname)
        strain_fname = write_dir / ("strain-" + fname)

        package = (str(witness_fname), str(strain_fname), strain)
        strain_q.put(package)

        # need to be able to os.stat witness fnames for latency
        with open(witness_fname, "w") as f:
            pass

    for i, update in enumerate(updates):
        package = Mock()
        package.x = update
        package.request_id = i
        writer._in_q.put({output_name: package})
    writer._in_q.put(None)

    while True:
        package = writer.get_package()
        if package is None:
            break
        writer.process(package)

    i = 0
    while True:
        try:
            fname, latency = writer._out_q.get_nowait()
        except Empty:
            break

        ts = TimeSeries.read(fname, channel=channel_name).value
        validate_frame(ts, i)
        i += 1

    assert i == (num_frames - 1)
