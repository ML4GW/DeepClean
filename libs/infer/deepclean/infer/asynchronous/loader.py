import time
from typing import Callable, Iterable, Optional, Union

import numpy as np
from gwpy.timeseries import TimeSeries, TimeSeriesDict
from hermes.stillwater import Package, PipelineProcess


# TODO: this should probably be a dedicated utility somewhere
# maybe in gwftools lib? Or just in the main part of this lib?
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


class FrameLoader(PipelineProcess):
    """Loads .gwf frames using filenames passed from upstream processes

    Loads gwf frames as gwpy TimeSeries dicts, then resamples
    and converts these frames to numpy arrays according to the
    order specified in `channels`.

    Args:
        inference_sampling_rate:
            How often kernels should be sampled from the timeseries
        step_size:
            The number of samples to take between returned chunks
        sample_rate:
            The rate at which to resample the loaded data
        sequence_id:
            A sequence id to assign to this stream of data.
            Unnecessary if not performing stateful streaming
            inference.
        preprocessor:
            A preprocessing function which will be applied
            to the data at load time after resampling. If
            `strain_q` is not `None`, this will _not_ be
            applied to the first channel in `channels`.
        strain_q:
            A queue in which to put the loaded filenames
            as well as the first channel of the resampled
            data. If left as None, a queue will be initialized
            internally.
        remove:
            Whether to remove frames from disk once they're read
    """

    def __init__(
        self,
        inference_sampling_rate: float,
        sample_rate: float,
        channels: Union[str, Iterable[str]],
        sequence_id: Optional[int] = None,
        preprocessor: Optional[Callable] = None,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        self.stride = int(sample_rate // inference_sampling_rate)
        self.sample_rate = sample_rate
        self.sequence_id = sequence_id
        self.channels = channels
        self.preprocessor = preprocessor

        self._idx = 0
        self._frame_idx = 0
        self._data = None
        self._end_next = False

    def get_package(self) -> Package:
        start = self._frame_idx * self.stride
        stop = start + self.stride

        # look ahead at whether we'll need to grab a frame
        # after this timestep so that we can see if this
        # will be the last step that we take
        # if sequence_end is False and therefore self._end_next
        # is False, we don't need to try to get another frame
        # since this will be the last one
        next_stop = stop + self.stride
        sequence_start = self._data is None
        sequence_end = self._end_next
        if (
            sequence_start or next_stop >= self._data.shape[1]
        ) and not sequence_end:
            # try to load in the next frame's worth of data
            try:
                fname = super().get_package()
            except StopIteration:
                # super().get_package() raised a StopIteration,
                # so catch it and indicate that this will be
                # the last inference that we'll produce
                if next_stop == self._data.shape[1]:
                    # if the next frame will end precisely at the
                    # end of our existing data, we'll have one more
                    # frame to process after this one, so set
                    # self._end_next to True
                    self._end_next = True
                else:
                    # otherwise, the next frame wouldn't be able
                    # to fit into the model input, so we're going
                    # to end here and just accept that we'll have
                    # some trailing data.
                    # TODO: should we append with zeros? How will
                    # this information get passed to whatever process
                    # is piecing information together at the other end?
                    sequence_end = True
            else:
                # otherwise append the new data to whatever
                # remaining data we have left to go through
                data = load_frame(fname, self.channels, self.sample_rate)
                if self._data is not None and start < self._data.shape[1]:
                    leftover = self._data[:, start:]
                    data = np.concatenate([leftover, data], axis=1)

                # reset everything
                self._data = data
                self._frame_idx = 0
                start, stop = 0, self.stride

        # create a package using the data, using
        # the internal index to set the request
        # id for downstream processes to reconstruct
        # the order in which data should be processed
        x = self._data[:, start:stop]
        package = Package(
            x=x,
            t0=time.time(),
            request_id=self._idx,
            sequence_id=self.sequence_id,
            sequence_start=sequence_start,
            sequence_end=sequence_end,
        )

        # increment the request index for the next request
        self._idx += 1
        self._frame_idx += 1

        return package

    def process(self, package):
        self.logger.debug(f"Submitting inference request {package.request_id}")
        super().process(package)

        # if this is the last package in the sequence,
        # raise a StopIteration so that downstream
        # processes know that we're done here
        if package.sequence_end:
            raise StopIteration
