import logging
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
    channels: Union[str, Iterable[str]],
    sample_rate: float,
    inference_sampling_rate: float,
) -> Package:
    stride = int(sample_rate / inference_sampling_rate)

    if isinstance(channels, str):
        data = np.array([])
    else:
        data = np.array([[] * len(channels)])
    slc = np.arange(stride)

    # manually iterate through the frame crawler so
    # we can intercept if it's about to run out and
    # mark the sequence as completed
    fname_it = iter(crawler)
    fname = next(fname_it)

    sequence_end = False
    while True:
        logging.debug(f"Loading frame file {fname}")
        frame = load_frame(fname, channels, sample_rate)
        data = np.append(data, frame, axis=-1)

        num_steps = (data.shape[-1] - 1) // stride
        for i in range(num_steps - 1):
            x = data.take(slc + i * stride, axis=-1)

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

            yield x, sequence_end

        # if we've run out of frames to load, kill the loop
        if sequence_end:
            break

        # slough off any already processed data along the time axis
        _, data = np.split(data, [(i + 1) * stride], axis=-1)
