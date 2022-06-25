import logging
import time
from typing import Iterable, Union

import numpy as np
from gwpy.timeseries import TimeSeries, TimeSeriesDict
from hermes.stillwater import Package


def load_frame(
    fname: str, channels: Union[str, Iterable[str]], sample_rate: float
) -> np.ndarray:
    """Load the indicated channels from the indicated frame file"""

    if isinstance(channels, str):
        # if we don't have multiple channels, then just grab the data
        data = TimeSeries.read(fname, channels)
        data.resample(sample_rate)
        data = data.value
    else:
        # otherwise stack the arrays
        data = TimeSeriesDict.read(fname, channels)
        data.resample(sample_rate)
        data = np.stack([data[i].value for i in channels])

    # return as the expected type
    return data.astype("float32")


def frame_iterator(
    crawler: Iterable,
    stride: int,
    channels: Union[str, Iterable[str]],
    sample_rate: float,
    sequence_id: int,
) -> Package:
    if isinstance(channels, str):
        data = np.array([])
        x = np.zeros((stride,))
    else:
        data = np.array([[] * len(channels)])
        x = np.zeros((len(channels), stride))
    slc = np.arange(stride)

    # manually iterate through the frame crawler so
    # we can intercept if it's about to run out and
    # mark the sequence as completed
    fname_it = iter(crawler)
    fname = next(fname_it)

    idx = 0
    sequence_end = False
    while True:
        logging.debug(f"Loading frame file {fname}")
        frame = load_frame(fname, channels, sample_rate)
        data = np.append(data, frame, axis=-1)

        num_steps = (data.shape[-1] - 1) // stride
        for i in range(num_steps - 1):
            # since we're buffering results to an output array
            # to save on data copies, we'll take 1 less stride
            # than we _might_ be able to in order to ensure that
            # the last stride isn't too short to be copied
            data.take(slc + i * stride, out=x, axis=-1)

            # build a package to send to downstream processes
            package = Package(
                x=x,
                t0=time.time(),
                request_id=idx,
                sequence_id=sequence_id,
                sequence_start=idx == 0,
                sequence_end=sequence_end,
            )

            # this basically checks if the next step will
            # be the last full stride of data we'll be able
            # to use. If it is, try loading a frame to see
            # if we have any left.
            if (i + 2) * stride > data.shape[-1]:
                try:
                    fname = next(fname_it)
                except StopIteration:
                    # we're out of frames _and_ the next
                    # step will be the last, so indicate
                    # that the sequence has terminated
                    sequence_end = True

            yield package
            idx += 1

        # if we've run out of frames to load, kill the loop
        if sequence_end:
            break

        # slough off any already processed data along the time axis
        _, data = np.split(data, [(i + 1) * stride], axis=-1)
