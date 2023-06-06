import time
from pathlib import Path
from queue import Empty, Queue
from typing import Iterable, List, Tuple

import numpy as np
from gwpy.timeseries import TimeSeries
from microservice.deployment import DataStream
from microservice.frames import frame_it


def strain_iterator(q: Queue):
    while True:
        try:
            response = q.get_nowait()
        except Empty:
            time.sleep(1e-3)
        else:
            if response is None:
                break
            yield response


def batch_iter(
    crawler: Iterable,
    channels: List[str],
    sample_rate: float,
    inference_sampling_rate: float,
    batch_size: int,
    q: Queue,
) -> Tuple[np.ndarray, bool]:
    """
    Iterate through the frame files returned by
    a crawler and load the relevant witness and
    strain data, placing the strain data into a
    `Queue` and breaking the witness timeseries
    up into `batch_size` updates of size
    `sample_rate // inference_sampling_rate` each.

    Arguments:
        crawler:
            Iterator which returns strain and
            witness frame file names
        channels:
            Channels to load from the frame files,
            with the strain channel name first
        sample_rate:
            Sample rate at which to resample all
            the loaded channels
        inference_sampling_rate:
            Rate at which to sample input windows to
            pass to DeepClean from the timeseries
        batch_size:
            Number of windows on which DeepClean will
            perform inference at once
        q:
            `Queue` object into which strain data and
            its associated filename will be placed
            for use by a cleaning callback thread.
    Returns:
        Array containing one batch worth of updates
            for streaming to the inference server
        Flag indicating whether this represents this
            batch represents the end of a sequence of
            updates, after which the associated
            snapshot state should be reset.
    """
    stride = int(batch_size * sample_rate / inference_sampling_rate)
    it = frame_it(crawler, channels, sample_rate)

    strain_fname, strain, witnesses = next(it)
    strain = TimeSeries(strain, channel=channels[0], sample_rate=sample_rate)
    q.put((strain, strain_fname))

    idx = 0
    while True:
        start = idx * stride
        stop = (idx + 1) * stride

        # check if the _next_ batch of updates
        # will require grabbing more data, that
        # way we can check if our data iterator
        # has been exhausted and can return True
        # for the sequence_end flag
        if (idx + 2) * stride > witnesses.shape[-1]:
            try:
                strain_fname, strain, update = next(it)
            except StopIteration:
                yield witnesses[:, start:stop], True
                q.put(None)
                break
            else:
                # put the strain data and its associated
                # filename in the queue that can be passed
                # to our callback thread
                strain = TimeSeries(
                    strain, channel=channels[0], sample_rate=sample_rate
                )
                q.put((strain, strain_fname))

                # slough off any old witness data that
                # we no longer need and append the new data
                witnesses = witnesses[:, start:]
                witnesses = np.concatenate([witnesses, update], axis=1)

                # reset all our indices to reflect the
                # new witness data array
                idx, start, stop = 0, 0, stride
        yield witnesses[:, start:stop], False
        idx += 1


def get_data_generators(
    data_dir: Path,
    ifo: str,
    channels: List[str],
    sample_rate: float,
    inference_sampling_rate: float,
    batch_size: int,
    start_first: bool = True,
    timeout: float = 10,
):
    t0 = 0 if start_first else -1
    stream = DataStream(data_dir, ifo)
    crawler = stream.crawl(t0, timeout)
    strain_q = Queue()
    witness_it = batch_iter(
        crawler,
        channels,
        sample_rate,
        inference_sampling_rate,
        batch_size,
        strain_q,
    )
    strain_it = strain_iterator(strain_q)
    return witness_it, strain_it
