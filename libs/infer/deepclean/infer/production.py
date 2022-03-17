import os
import pickle
import time
from queue import Empty, Queue
from typing import Optional

import numpy as np
from gwpy.timeseries import TimeSeries
from hermes.gwftools.gwftools import _parse_frame_name
from hermes.stillwater import PipelineProcess


class FrameWriter(PipelineProcess):
    def __init__(
        self,
        write_dir: str,
        channel_name: str,
        inference_sampling_rate: float,
        sample_rate: float,
        postprocess_pkl: str,
        strain_q: Queue,
        strain_timeout: Optional[float] = None,
        memory: float = 10,
        look_ahead: float = 0.05,
        throw_away: Optional[int] = None,
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
            postprocess_pkl:
                The path to a pickle file containing a serialized
                `deepclean.signal.op.Op` to apply on the aggregated
                noise predictions (including the past and future data
                indicated by the `memory` and `look_ahead` arguments)
                before subtraction from the strain channel. Will be
                applied with `inverse=True`.
            strain_q:
                A queue from which strain data will be retrieved from
                upstream processes
            strain_timeout:
                The number of seconds to wait without a new strain file
                before raising an error. If left as `None`, process will
                wait indefinitely for each strain file to arrive in
                `strain_q`.
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
            output_name:
                The name given by the inference server to the streaming
                DeepClean output tensor.
        """
        super().__init__(*args, **kwargs)

        # make the write directory if it doesn't exist
        os.makedirs(write_dir, exist_ok=True)

        # record some of our writing parameters as-is
        self.write_dir = write_dir
        self.channel_name = channel_name
        self.sample_rate = sample_rate

        # record some of the parameters of the data,
        # mapping from time or frequency units to
        # sample units
        self.step_size = int(1 / inference_sampling_rate)
        self.memory = int(memory * sample_rate)
        self.look_ahead = int(look_ahead * sample_rate)
        self.throw_away = throw_away

        # record properties related to how we're going
        # to get the strain data
        self.strain_timeout = strain_timeout
        self.strain_q = strain_q

        with open(postprocess_pkl, "rb") as f:
            self.postprocessor = pickle.load(f)

        self._strains = []
        self._noises = np.array([])
        self._covered_idx = np.array([])
        self._frame_idx = 0
        self._thrown_away = 0

    def get_package(self):
        # now get the next inferred noise estimate
        noise_prediction = super().get_package()

        # first see if we have any new strain data to collect
        try:
            if len(self._noises) == 0:
                fname, strain = self.strain_q.get(True, 10)
            else:
                fname, strain = self.strain_q.get(False)
        except Empty:
            if len(self._noises) == 0:
                raise RuntimeError("No strain data after 10 seconds")
        else:
            # if we do, add it to our running list of strains
            self.logger.debug(f"Adding strain file {fname} to strains")
            self._strains.append((fname, strain))

            # create a blank array to fill out our noise
            # and idx arrays as we collect more data
            zeros = np.zeros_like(strain)
            self._noises = np.append(self._noises, zeros)
            self._covered_idx = np.append(self._covered_idx, zeros)

        return noise_prediction

    def process(self, package):
        # grab the noise prediction from the package
        # slice out the batch and channel dimensions,
        # which will both just be 1 for this pipeline
        package = package["aggregator"]
        self.logger.debug(
            "Received response for package {}".format(package.request_id)
        )
        x = package.x.reshape(-1)
        if len(x) != self.step_size:
            raise ValueError(
                "Noise prediction is of wrong length {}".format(len(x))
            )

        if self.throw_away is not None and self._thrown_away < self.throw_away:
            self._thrown_away += 1
            self.logger.debug(
                "Throwing away response for package {}".format(
                    package.request_id
                )
            )
            if self._thrown_away == self.throw_away:
                self.logger.debug("Done with throwaway responses")
            return

        # use the package request id to figure out where
        # in the blank noise array we need to insert
        # this prediction. Subtract the running index
        # of the total number of samples we've processed so far
        offset = max(self._frame_idx - self.past_samples, 0)
        start = int(package.request_id * self.step_size - offset)
        if self.throw_away is not None:
            start -= int(self.throw_away * self.step_size)
        self._noises[start : start + len(x)] = x

        # update our mask to indicate which parts of our
        # noise array have had inference performed on them
        self._covered_idx[start : start + len(x)] = 1
        if len(self._strains) == 0:
            self.logger.debug("No strains to clean, continuing")
            return

        # if we've completely performed inference on
        # an entire frame's worth of data, postprocess
        # the predictions and produce the cleaned estimate
        past_samples = min(self.past_samples, self._frame_idx)
        limit = past_samples + len(self._strains[0][1]) + self.future_samples
        if self._covered_idx[:limit].all():
            # pop out the earliest strain and filename and
            (witness_fname, strain_fname), strain = self._strains.pop(0)
            self.logger.debug("Cleaning strain file " + strain_fname)
            fname = os.path.basename(strain_fname)
            timestamp, _ = _parse_frame_name(fname)

            # now postprocess the noise and strain channels
            noise = self.preprocessor.uncenter(self._noises)
            noise = self.preprocessor.filter(noise)
            noise = noise[past_samples : past_samples + len(strain)]

            self._frame_idx += len(strain)
            if (
                self._covered_idx.sum() - self.future_samples
            ) > self.past_samples:
                self._noises = self._noises[len(strain) :]
                self._covered_idx = self._covered_idx[len(strain) :]

            # remove the noise from the strain channel and
            # use it to create a timeseries we can write to .gwf
            cleaned = strain - noise
            timeseries = TimeSeries(
                cleaned,
                t0=timestamp,
                sample_rate=self.sample_rate,
                channel=self.channel_name,
            )

            # write the file and pass the written filename
            # to downstream processess
            write_fname = os.path.join(self.write_dir, fname)
            timeseries.write(write_fname)
            latency = time.time() - os.stat(witness_fname).st_mtime
            super().process((write_fname, latency))
