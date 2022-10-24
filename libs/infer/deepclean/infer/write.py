import logging
import math
import time
from pathlib import Path
from typing import Callable, Iterator, Optional, Tuple

import numpy as np
from gwpy.timeseries import TimeSeries

from deepclean.gwftools.frames import parse_frame_name


class FrameWriter:
    def __init__(
        self,
        write_dir: Path,
        strain_iter: Iterator,
        channel_name: str,
        inference_sampling_rate: float,
        sample_rate: float,
        frame_length: int = 1,
        batch_size: float = 1,
        postprocessor: Optional[Callable] = None,
        memory: float = 10,
        look_ahead: float = 0.05,
        aggregation_steps: int = 0,
    ) -> None:
        """DeepClean predictions postprocessor and frame writer

        Asynchronous process for concatenating averaged
        responses from a streaming DeepClean model, post-
        processing them, subtacting them from the associated
        strain data, then writing the cleaned strain to a
        frame file. Frame files will be written using the
        same filename given to the associated raw strain file.

        Args:
            write_dir:
                The directory to which to save the cleaned strain frames
            channel_name:
                The name to give the strain channel in the cleaned frames
            step_size:
                The number of samples to expect in responses streamed
                back from the inference server
            inference_sampling_rate:
                The rate at which kernels are sampled from the strain
                and witness timeseries
            memory:
                The number of seconds of noise predictions from
                _before_ each frame being cleaned to include in-memory
                when performing postprocessing. Particularly useful
                for the purposes of avoiding filtering edge effects
            look_ahead:
                The number of seconds of noise predictions from
                _after_ each frame being cleaned to wait for before
                performing postprocessing. Note that this introduces
                extra latency on top of whatever latency is incurred
                by aggregation on the inference server.
            aggregation_steps:
                The number of overlapping timesteps over which overlapping
                segments are averaged on the inference server
        """

        # make the write directory if it doesn't exist
        write_dir.mkdir(parents=True, exist_ok=True)

        # record some of our writing parameters as-is
        self.write_dir = write_dir
        self.strain_iter = strain_iter
        self.sample_rate = sample_rate
        self.channel_name = channel_name

        # record some of the parameters of the data, mapping
        # from time or frequency units to sample units
        self.frame_size = int(sample_rate * frame_length)
        self.stride = int(sample_rate // inference_sampling_rate)
        self.steps_per_frame = math.ceil(self.frame_size / self.stride)

        self.memory = int(memory * sample_rate)
        self.samples_ahead = int(look_ahead * sample_rate)
        self.steps_ahead = math.ceil(self.samples_ahead / self.stride)

        self.batch_size = batch_size
        self.aggregation_steps = aggregation_steps
        self.agg_batches, self.agg_leftover = divmod(
            aggregation_steps, batch_size
        )

        # load in our postprocessing pipeline
        self.postprocessor = postprocessor

        self._zeros = np.zeros((self.frame_size,), dtype=np.float32)
        self._noise = np.array([])
        self._frame_idx = 0
        self._latest_seen = -1

        self.logger = logging.getLogger("Frame Writer")

    def block(self, i: int, callback: Optional[Callable] = None):
        while self._latest_seen < i:
            if callback is not None:
                callback()
            time.sleep(1e-3)

    def validate_response(
        self, noise_prediction: np.ndarray, request_id: int
    ) -> Tuple[np.ndarray, int]:
        """
        Parse the response from the server to get
        the noise prediction and the corresponding
        request id. Ensure that the request id falls
        in the expected order and if not, log the ids
        that we missed.

        Returns:
            The numpy array of the server response
        """
        self.logger.debug(f"Received response for package {request_id}")

        # flatten out to 1D and verify that this
        # output aligns with our expectations
        x = noise_prediction.reshape(-1)
        if len(x) != (self.stride * self.batch_size):
            raise ValueError(
                "Noise prediction is of wrong length {}, "
                "expected length {}".format(len(x), self.stride)
            )

        # now make sure that this request is arriving in order
        if request_id < self._latest_seen:
            self.logger.warning(f"Request id {request_id} came in late")
        else:
            if request_id > (self._latest_seen + 1):
                # if we skipped some requests, log the ones we missed
                for i in range(1, request_id - self._latest_seen):
                    idx = self._latest_seen + i
                    self.logger.warning(f"No response for request id {idx}")

            # update `self._latest_seen` to reflect whatever
            # the new latest request id is
            self._latest_seen = request_id

        if request_id < self.agg_batches:
            # throw away the first `aggregation_steps` responses, since
            # these technically corresponds to predictions from the _past_.
            # Return `None`s so that we know not to update our internal arrays
            self.logger.debug(
                f"Throwing away received package id {request_id}"
            )
            return None
        elif request_id == self.agg_batches:
            # if our batch size doesn't evenly divide the number
            # of aggregation steps, slice off any steps in this
            # batch of responses that are aggregation steps
            x = x[self.agg_leftover * self.stride :]

        # otherwise return the parsed array and its request id
        return x

    def update_prediction_array(self, x: np.ndarray, request_id: int) -> int:
        """
        Update our initialized `_noise` array with the prediction
        `x` by inserting it at the appropriate location. Update
        our `_mask` array to indicate that this stretch has been
        appropriately filled.
        """

        # use the package request id to figure out where
        # in the blank noise array we need to insert
        # this prediction. Subtract the steps that we threw away
        total_steps = request_id * self.batch_size
        step_idx = total_steps - self.aggregation_steps
        start_idx = 0 if step_idx < 0 else int(step_idx * self.stride)

        # If we've sloughed off any past data so far,
        # make sure to account for that
        frame_idx = self._frame_idx * self.frame_size
        start_idx -= max(frame_idx - self.memory, 0)

        # now make sure that we have data to fill out,
        # otherwise extend the array
        if (start_idx + len(x)) > len(self._noise):
            self._noise = np.append(self._noise, self._zeros)

        # now insert the response into the existing array
        # TODO: should we check that this is all 0s to
        # double check ourselves here?
        self._noise[start_idx : start_idx + len(x)] = x

        # return the number of steps that have been completed
        return step_idx + self.batch_size

    def clean(self, noise: np.ndarray) -> Tuple[Path, float]:
        """
        Grab a new strain file, use the provided timeseries of
        noise predictions to clean it, then write the cleaned
        strain to a .gwf file. Return the name of this file as
        well as the time delta in seconds between when the
        strain file was written and when the cleaned strain file
        completes writing for profiling purposes.

        Args:
            noise:
                The timeseries of DeepClean noise predictions,
                including any past and future data needed for
                postprocessing before subtraction.
        Returns:
            The name of the file the cleaned strain was written to
        """

        # apply any postprocessing
        # to the padded noise channel,
        # then slice out current frame
        if self.postprocessor is not None:
            noise = self.postprocessor(noise)
        offset = -self.samples_ahead
        noise_segment = noise[offset - self.frame_size : offset]

        # now grab the next strain segment
        # and subtract this noise from it
        strain, fname = next(self.strain_iter)
        strain = strain - noise_segment

        # create a timeseries from the cleaned strain that
        # we can write to a .gwf. Make sure to give it
        # the same timestamp as the original file
        timestamp = parse_frame_name(fname.name)[1]
        timeseries = TimeSeries(
            strain,
            t0=timestamp,
            sample_rate=self.sample_rate,
            channel=self.channel_name + "-CLEANED",
        )

        # write the timeseries to our `write_dir`,
        # measure the latency from when the witness
        # file became available and when we wrote the
        # frame file for profiling purposes
        write_path = self.write_dir / fname.name
        timeseries.write(write_path)

        # if moving this frame into our memory will
        # give us more than `memory` samples, then
        # slough off a frame from the start of our
        # _noises and _mask arrays
        extra = len(noise) - self.samples_ahead - self.memory
        if extra > 0:
            self._noise = self._noise[extra:]
        return write_path

    def __call__(self, noise_prediction: np.ndarray, request_id: int, *arg):
        # get the server response and corresponding
        # request id from the passed package
        noise_prediction = self.validate_response(noise_prediction, request_id)
        if noise_prediction is None:
            return

        # now insert it into our `_noises` prediction
        # array and update our `_mask` to reflect this
        step_idx = self.update_prediction_array(noise_prediction, request_id)

        # modulate the total number of cleaning steps taken by the
        # number of steps in each frame to see if the current
        # cleanable index is past the end of the next frame
        div, rem = divmod(step_idx, self.steps_per_frame)
        if div <= self._frame_idx:
            return None
        elif rem >= self.steps_ahead or div > self._frame_idx + 1:
            # slice out the data corresponding to current
            # frame, first figuring out if we have a full
            # memory to use or not
            frame_idx = self._frame_idx * self.frame_size
            idx = min(self.memory, frame_idx)
            idx += self.frame_size + self.samples_ahead
            noise = self._noise[:idx]

            fname = self.clean(noise)
            self.logger.info(f"Wrote cleaned frame {fname}")

            self._frame_idx += 1
            return fname
        else:
            return None
