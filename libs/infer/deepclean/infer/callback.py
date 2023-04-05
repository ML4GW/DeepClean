import math
import time
from typing import Callable, Optional

import numpy as np

from deepclean.logging import logger


class State:
    def __init__(
        self,
        name: str,
        frame_length: float,
        memory: float,
        filter_pad: float,
        sample_rate: float,
        inference_sampling_rate: float,
        batch_size: int,
        aggregation_steps: int
    ) -> None:
        self.frame_size = int(sample_rate * frame_length)
        self.stride = int(sample_rate // inference_sampling_rate)
        self.steps_per_frame = math.ceil(self.frame_size / self.stride)

        self.memory = int(memory * sample_rate)
        self.samples_ahead = int(filter_pad * sample_rate)
        self.steps_ahead = math.ceil(self.samples_ahead / self.stride)

        self.batch_size = batch_size
        self.aggregation_steps = aggregation_steps
        self.agg_batches, self.agg_leftover = divmod(
            aggregation_steps, batch_size
        )

        self._zeros = np.zeros((self.frame_size,), dtype=np.float32)
        self._state = np.zeros((0,), dtype=np.float32)

        self._frame_idx = 0
        self._latest_seen = -1
        self.logger = logger.get_logger(f"Output state {name}")

    def validate(self, response, request_id):
        # flatten out to 1D and verify that this
        # output aligns with our expectations
        x = response.reshape(-1)
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

    def update(self, x, request_id):
        x = self.validate(x, request_id)
        if x is None:
            return None

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
        if (start_idx + len(x)) > len(self._state):
            self._state = np.append(self._state, self._zeros)

        # now insert the response into the existing array
        # TODO: should we check that this is all 0s to
        # double check ourselves here?
        self._state[start_idx : start_idx + len(x)] = x

        # return the number of steps that have been completed
        step_idx = step_idx + self.batch_size

        # modulate the total number of cleaning steps taken by the
        # number of steps in each frame to see if the current
        # cleanable index is past the end of the next frame
        div, rem = divmod(step_idx, self.steps_per_frame)
        if (
            (div == (self._frame_idx + 1) and rem >= self.steps_ahead)
            or div > self._frame_idx + 1
        ):
            frame_idx = self._frame_idx * self.frame_size
            idx = min(self.memory, frame_idx)
            idx += self.frame_size + self.samples_ahead
            noise = self._state[:idx]

            extra = len(noise) - self.samples_ahead - self.memory
            if extra > 0:
                self._state = self._state[extra:]
            self._frame_idx += 1
            return noise
        return None


class Callback:
    def __init__(
        self, postprocessor: Callable, **states: State
    ):
        self.postprocessor = postprocessor
        self.states = states
        self.logger = logger.get_logger("Infer callback")
        self.logger.debug("In callback")

    def block(self, i: int = 0, callback: Optional[Callable] = None):
        while any([state._latest_seen < i for state in self.states.values()]):
            if callback is not None:
                response = callback()
                if response is not None:
                    return response

    def __call__(self, responses: np.ndarray, request_id: int, *arg):
        predictions = {}
        for state_name, y in responses.items():
            state = self.states[state_name]
            prediction = state.update(y, request_id)
            if prediction is not None:
                predictions[state_name] = prediction

        if predictions:
            self.logger.debug(f"Cleaning frame {state._frame_idx - 1}")
            return self.postprocessor(**predictions)
        return None
