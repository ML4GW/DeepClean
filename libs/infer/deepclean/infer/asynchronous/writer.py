import os
import pickle
import time
from queue import Empty, Queue
from typing import Optional, Tuple

import numpy as np
from gwpy.timeseries import TimeSeries
from hermes.stillwater import PipelineProcess


def _parse_frame_name(fname: str) -> Tuple[int, int]:
    """Use the name of a frame file to infer its initial timestamp and length

    Copied from gw-iaas/libs/hermes/gwftools for now
    in order to avoid it as a dependency.

    Expects frame names to follow a standard nomenclature
    where the name of the frame file ends {timestamp}-{length}.gwf

    Args:
        fname:
            The name of the frame file
    Returns
        The initial GPS timestamp of the frame file
        The length of the frame file in seconds
    """

    fname = fname.replace(".gwf", "")
    timestamp, length = tuple(map(int, fname.split("-")[-2:]))
    return timestamp, length


class FrameWriter(PipelineProcess):
    def __init__(
        self,
        write_dir: str,
        channel_name: str,
        inference_sampling_rate: float,
        sample_rate: float,
        strain_q: Queue,
        postprocess_pkl: Optional[str] = None,
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
            strain_q:
                A queue from which strain data will be retrieved from
                upstream processes
            postprocess_pkl:
                The path to a pickle file containing a serialized
                `deepclean.signal.op.Op` to apply on the aggregated
                noise predictions (including the past and future data
                indicated by the `memory` and `look_ahead` arguments)
                before subtraction from the strain channel. Will be
                applied with `inverse=True`. If left as `None`, no
                postprocessing will be applied.
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

        # make the write directory if it doesn't exist
        os.makedirs(write_dir, exist_ok=True)

        # record some of our writing parameters as-is
        self.write_dir = write_dir
        self.channel_name = channel_name
        self.sample_rate = sample_rate
        self.strain_q = strain_q

        # record some of the parameters of the data,
        # mapping from time or frequency units to
        # sample units
        self.step_size = int(1 / inference_sampling_rate)
        self.memory = int(memory * sample_rate)
        self.look_ahead = int(look_ahead * sample_rate)
        self.aggregation_steps = aggregation_steps

        # load in our postprocessing pipeline
        if postprocess_pkl is None:
            with open(postprocess_pkl, "rb") as f:
                self.postprocessor = pickle.load(f)
        else:
            self.postprocessor = None

        # next we'll initialize a bunch of arrays
        # and indices we'll need to do efficient
        # asynchronous processing of responses

        # _strains will contain a list of the strain frames
        # that we've retrieved from `self.strain_q` but which
        # haven't been cleaned yet
        self._strains = []

        # _noises will contain a pre-initialized array of all
        # the noise predictions we _expect_ to get given the
        # number of oustanding strains we have (plus any past
        # and future data we want to keep for postprocessing
        # purposes), which we'll populate with responses as
        # they come in
        self._noises = np.array([])

        # _mask will be the same size as _noises and contain
        # a boolean mask indicating which indices of _noises
        # have been populated by a response
        self._mask = np.array([])

        # _frame_idx keeps track of where in our "infinite"
        # timeseries the first sample of the current _noises
        # array falls
        self._frame_idx = 0

        # _thrown away keeps track of how many initial
        # responses we've ignored to account for aggregation
        # on the server side
        self._thrown_away = 0

    def get_package(self):
        # now get the next inferred noise estimate
        noise_prediction = super().get_package()

        # first see if we have any new strain data to collect
        try:
            # if we haven't begun yet, give ourselves a few
            # seconds to wait for strain data to come in.
            # If it doesn't, then assume something has gone wrong
            if len(self._noises) == 0:
                fname, strain = self.strain_q.get(True, 10)
            else:
                # otherwise, just immediately raise an error
                # that we'll ignore if there's nothing there
                fname, strain = self.strain_q.get(False)
        except Empty:
            if len(self._noises) == 0:
                # if we timed out because no data has arrived
                # yet, assume something went wrong and raise
                raise RuntimeError("No strain data after 10 seconds")
        else:
            # if we there was a new strain frame available,
            # add it to our running list of strains to clean.
            # include the filename for metadata purposes
            self.logger.debug(f"Adding strain file {fname} to strains")
            self._strains.append((fname, strain))

            # add blank arrays to our existing _noises
            # and _mask arrays accounting for the noise
            # predictions we expect to eventually use to
            # clean the strain we just grabbed
            zeros = np.zeros_like(strain)
            self._noises = np.append(self._noises, zeros)
            self._covered_idx = np.append(self._covered_idx, zeros)

        # return the noise prediction for processing
        # in `self.process`
        return noise_prediction

    def update_prediction_array(self, x: np.ndarray, request_id: int) -> None:
        """
        Update our initialized `_noises` array with the prediction
        `x` by inserting it at the appropriate location. Update
        our `_mask` array to indicate that this stretch has been
        appropriately filled.
        """

        # throw away the first `aggregation_steps` responses
        # since these technically corresponds to predictions
        # from the _past_
        if self._thrown_away < self.aggregation_steps:
            self._thrown_away += 1
            self.logger.debug(
                f"Throwing away received package id {request_id}"
            )

            if self._thrown_away == self.aggregation_steps:
                self.logger.debug(
                    "Outputs have caught up with start of "
                    f"timeseries after {self._thrown_away} "
                    "responses. Done with throwing away responses."
                )

            # don't update any our arrays with this response
            return

        # use the package request id to figure out where
        # in the blank noise array we need to insert
        # this prediction. Subtract the steps that we threw away
        start_id = request_id - self.aggregation_steps
        start_idx = int(start_id * self.step_size)

        # If we've sloughed off any past data so far,
        # make sure to account for that
        start_idx -= max(self._frame_idx - self.past_samples, 0)

        # now insert the response into the existing array
        # TODO: should we check that this is all 0s to
        # double check ourselves here?
        self._noises[start_idx : start_idx + len(x)] = x

        # update our mask to indicate which parts of our
        # noise array have had inference performed on them
        self._mask[start_idx : start_idx + len(x)] = 1

    def clean(self, noise: np.ndarray):
        # pop out the earliest strain data from our
        # _strains tracker
        (witness_fname, strain_fname), strain = self._strains.pop(0)
        self.logger.debug("Cleaning strain file " + strain_fname)

        # now postprocess the noise channel and
        # slice off the relevant frame to subtract
        # from the strain channel
        if self.postprocessor is not None:
            noise = self.postprocessor(noise, inverse=True)
        noise = noise[-self.look_ahead - len(strain) : -self.look_ahead]
        cleaned = strain - noise

        # increment our _frame_idx to account for
        # the frame we're about to write
        self._frame_idx += len(strain)

        # create a timeseries from the cleaned strain that
        # we can write to a `.gwf`. Make sure to give it
        # the same timestamp as the original file
        fname = os.path.basename(strain_fname)
        timestamp, _ = _parse_frame_name(fname)
        timeseries = TimeSeries(
            cleaned,
            t0=timestamp,
            sample_rate=self.sample_rate,
            channel=self.channel_name,
        )

        # write the timeseries to our `write_dir`,
        # measure the latency from when the witness
        # file became available and when we wrote the
        # frame file for profiling purposes
        write_path = os.path.join(self.write_dir, fname)
        timeseries.write(write_path)
        latency = time.time() - os.stat(witness_fname).st_mtime

        # if moving this frame into our memory will
        # give us more than `memory` samples, then
        # slough off a frame from the start of our
        # _noises and _mask arrays
        if len(noise) - self.look_ahead > self.memory:
            self._noises = self._noises[len(strain) :]
            self._mask = self._mask[len(strain) :]

        return write_path, latency

    def process(self, package: dict):
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
        else:
            self.logger.debug(
                f"Received response for package {package.request_id}"
            )

        # flatten out to 1D and verify that this
        # output aligns with our expectations
        x = package.x.reshape(-1)
        if len(x) != self.step_size:
            raise ValueError(
                "Noise prediction is of wrong length {}".format(len(x))
            )

        # now insert it into our `_noises` prediction
        # array and update our `_mask` to reflect this
        self.update_prediction_array(x, package.request_id)

        # if we don't have any strain frames to process,
        # then we have nothing left to do
        if len(self._strains) == 0:
            self.logger.debug("No strains to clean, continuing")
            return

        # if we've completely performed inference on
        # an entire frame's worth of data, postprocess
        # the predictions and produce the cleaned estimate

        # if we haven't accumulated enough frames to keep
        # a full memory, then use what we have
        memory = min(self.memory, self._frame_idx)

        # make sure that the full memory, the current
        # frame, and the any future samples are all
        # accounted for
        limit = memory + len(self._strains[0][1]) + self.look_ahead
        if self._mask[:limit].all():
            # clean the current frame and write it to
            # our `write_dir`. Get the name of the file
            # we wrote it to as well as the incurred latency
            fname, latency = self.clean(self._noises[:limit])

            # push these to our `out_q` for use by downstream processes
            super().process((fname, latency))