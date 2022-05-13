import time
from pathlib import Path
from typing import Callable, Optional

import numpy as np
from gwpy.timeseries import TimeSeries
from hermes.stillwater import PipelineProcess

from deepclean.gwftools.frames import parse_frame_name
from deepclean.infer.asynchronous.loader import load_frame
from deepclean.infer.frame_crawler import FrameCrawler


class FrameWriter(PipelineProcess):
    def __init__(
        self,
        data_dir: Path,
        write_dir: Path,
        channel_name: str,
        inference_sampling_rate: float,
        sample_rate: float,
        t0: Optional[float] = None,
        postprocessor: Optional[Callable] = None,
        memory: float = 10,
        look_ahead: float = 0.05,
        aggregation_steps: int = 0,
        output_name: str = "aggregator",
        *args,
        **kwargs,
    ) -> None:
        """Asynchronous DeepClean predictions postprocessor and frame writer

        Asynchronous process for concatenating averaged
        responses from a streaming DeepClean model, post-
        processing them, subtacting them from the associated
        strain data, then writing the cleaned strain to a
        frame file. Frame files will be written using the
        same filename given to the associated raw strain file.

        Args:
            data_dir:
                The directory from which to read strain files to clean
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
            output_name:
                The name given by the inference server to the streaming
                DeepClean output tensor.
        """
        super().__init__(*args, **kwargs)
        self.crawler = FrameCrawler(data_dir, t0, timeout=1)

        # make the write directory if it doesn't exist
        write_dir.mkdir(parents=True, exist_ok=True)

        # record some of our writing parameters as-is
        self.write_dir = write_dir
        self.sample_rate = sample_rate
        self.output_name = output_name
        self.channel_name = channel_name

        # record some of the parameters of the data, mapping
        # from time or frequency units to sample units
        self.memory = int(memory * sample_rate)
        self.look_ahead = int(look_ahead * sample_rate)
        self.aggregation_steps = aggregation_steps

        self.frame_size = int(sample_rate * self.crawler.length)
        self.stride = int(sample_rate // inference_sampling_rate)
        self.steps_per_frame = self.frame_size // self.stride

        # load in our postprocessing pipeline
        self.postprocessor = postprocessor

        self._zeros = np.zeros((self.frame_size,), dtype=np.float32)
        self._noise = np.array([])
        self._frame_idx = 0
        self._latest_seen = -1

    def process_package(self, package: dict):
        # grab the noise prediction from the package
        # slice out the batch and channel dimensions,
        # which will both just be 1 for this pipeline
        try:
            package = package[self.output_name]
        except KeyError:
            raise ValueError(
                "No output named {} returned by server, "
                "available tensors are {}".format(
                    self.output_name, list(package.keys())
                )
            )

        request_id = package.request_id
        self.logger.debug(f"Received response for package {request_id}")

        # flatten out to 1D and verify that this
        # output aligns with our expectations
        x = package.x.reshape(-1)
        if len(x) != self.stride:
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

            self._latest_seen = request_id

        # throw away the first `aggregation_steps` responses
        # since these technically corresponds to predictions
        # from the _past_
        if request_id < self.aggregation_steps:
            self.logger.debug(
                f"Throwing away received package id {request_id}"
            )
            # don't update any our arrays with this response
            return None, None
        return x, request_id

    def update_prediction_array(self, x: np.ndarray, request_id: int) -> None:
        """
        Update our initialized `_noise` array with the prediction
        `x` by inserting it at the appropriate location. Update
        our `_mask` array to indicate that this stretch has been
        appropriately filled.
        """

        # use the package request id to figure out where
        # in the blank noise array we need to insert
        # this prediction. Subtract the steps that we threw away
        start_id = request_id - self.aggregation_steps
        start_idx = int(start_id * self.stride)

        # If we've sloughed off any past data so far,
        # make sure to account for that
        start_idx -= max(self._frame_idx - self.memory, 0)

        # now make sure that we have data to fill out,
        # otherwise extend the array
        if (start_idx + len(x)) > len(self._noise):
            self._noise = np.append(self._noise, self._zeros)

        # now insert the response into the existing array
        # TODO: should we check that this is all 0s to
        # double check ourselves here?
        self._noise[start_idx : start_idx + len(x)] = x

    def clean(self, noise: np.ndarray):
        strain_fname = next(self.crawler)
        self.logger.debug(f"Cleaning strain file {strain_fname}")

        strain = load_frame(strain_fname, self.channel_name, self.sample_rate)
        if len(strain) != self.frame_size:
            raise ValueError(
                "Strain from file {} has unexpected length {}".format(
                    strain_fname, len(strain)
                )
            )

        # apply any postprocessing to the noise channel
        if self.postprocessor is not None:
            noise = self.postprocessor(noise)

        # now slice out just the segment we're concerned with
        # and subtract it from the strain channel
        start = -len(strain) - self.look_ahead
        stop = -self.look_ahead
        noise_segment = noise[start:stop]
        strain = strain - noise_segment

        # create a timeseries from the cleaned strain that
        # we can write to a .gwf. Make sure to give it
        # the same timestamp as the original file
        timestamp = parse_frame_name(strain_fname.name)[1]
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
        write_path = self.write_dir / strain_fname.name
        timeseries.write(write_path)
        latency = time.time() - strain_fname.stat().st_mtime

        # if moving this frame into our memory will
        # give us more than `memory` samples, then
        # slough off a frame from the start of our
        # _noises and _mask arrays
        extra = len(noise) - self.look_ahead - self.memory
        if extra > 0:
            self._noise = self._noise[extra:]

        # increment our _frame_idx to account for
        # the frame we're about to write
        self._frame_idx += len(strain)

        return write_path, latency

    def process(self, package: dict):
        # get the server response and corresponding
        # request id from the passed package
        response, request_id = self.process_package(package)
        if response is None:
            return

        # now insert it into our `_noises` prediction
        # array and update our `_mask` to reflect this
        self.update_prediction_array(response, request_id)

        # TODO: this won't generalize to strides that
        # aren't a factor of the frame size, which doesn't
        # feel like an insane constraint. Should this be
        # enforced explicitly somewhere?
        steps_ahead = (self.look_ahead - 1) // self.stride + 1
        step_idx = request_id - steps_ahead - self.aggregation_steps
        div, rem = divmod(step_idx, self.steps_per_frame)
        if div > 0 and rem == 0:
            idx = min(self.memory, self._frame_idx)
            idx += self.frame_size + self.look_ahead
            noise = self._noise[:idx]

            fname, latency = self.clean(noise)
            super().process((fname, latency))
