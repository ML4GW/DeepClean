import logging
import time
from contextlib import contextmanager
from typing import Callable

import numpy as np
import tritonclient.grpc as triton


class SimpleCallback:
    def __init__(self, name: str):
        self.name = name

        self.max_id_seen = -1
        self.predictions = np.array([])

        self.stopped = False
        self.error = None

    def __call__(self, result, error=None):
        if error is not None and not self.stopped:
            self.stopped = True
            self.error = str(error)
            logging.exception(f"Encountered error in callback: {error}")
        elif self.stopped:
            return

        request_id = int(result.get_response().id)
        logging.debug(f"Received response for request id {request_id}")

        y = result.as_numpy(self.name)[0]
        diff = request_id - self.max_id_seen - 1

        if diff > 0:
            zeros = np.zeros((diff * len(y),))
            self.predictions = np.concatenate([self.predictions, zeros, y])
            self.max_id_seen = request_id
        elif diff < 0:
            self.predictions[diff * len(y) : (diff + 1) * len(y)] = y
        else:
            self.predictions = np.append(self.predictions, y)
            self.max_id_seen += 1
        logging.debug(
            "Predicion array now {} samples long".format(len(self.predictions))
        )


@contextmanager
def begin_inference(
    client: triton.InferenceServerClient,
    model_name: str,
):
    metadata = client.get_model_metadata(model_name)
    output_name = metadata.outputs[0].name
    callback = SimpleCallback(output_name)

    input = triton.InferInput(
        name=metadata.inputs[0].name,
        shape=metadata.inputs[0].shape,
        datatype=metadata.inputs[0].datatype,
    )
    with client:
        client.start_stream(callback=callback)
        yield input, callback


def submit_for_inference(
    client: triton.InferenceServerClient,
    input: triton.InferInput,
    X: np.ndarray,
    stride: int,
    initial_request_id: int,
    sequence_id: int = 1001,
    model_name: str = "deepclean-stream",
    model_version: int = 1,
    sequence_end: bool = False,
) -> None:
    num_updates = (X.shape[-1] - 1) // stride + 1
    for i in range(num_updates):
        x = X[:, i * stride : (i + 1) * stride]
        input.set_data_from_numpy(x[None])

        request_id = initial_request_id + i
        logging.debug(
            f"Submitting inference request for request id {request_id}"
        )
        client.async_stream_infer(
            model_name,
            model_version=str(model_version),
            inputs=[input],
            sequence_id=sequence_id,
            request_id=str(request_id),
            sequence_start=(initial_request_id == 0) & (i == 0),
            sequence_end=sequence_end & (i == (num_updates - 1)),
        )
        time.sleep(1.5e-3)

    if (i + 1) * stride < X.shape[-1]:
        remainder = X[:, (i + 1) * stride :]
    else:
        remainder = None
    return remainder, request_id + 1


def online_postprocess(
    predictions: np.ndarray,
    strain: np.ndarray,
    frame_length: float,
    postprocessor: Callable,
    filter_memory: float,
    filter_lead_time: float,
    sample_rate: float,
):
    assert len(predictions) == len(strain)

    frame_size = int(sample_rate * frame_length)
    memory_size = int(sample_rate * filter_memory)
    lead_size = int(sample_rate * filter_lead_time)

    # cut off the last frame because we won't have
    # any lead time for it
    num_frames = (len(predictions) - 1) // frame_size
    frames = []
    for i in range(num_frames):
        start = max(i * frame_size - memory_size, 0)
        stop = (i + 1) * frame_size + lead_size

        prediction = predictions[start:stop]
        prediction = postprocessor(prediction, inverse=True)

        prediction = prediction[-frame_size - lead_size: -lead_size]
        target = strain[stop - frame_size - lead_size : stop - lead_size]

        clean = target - prediction
        frames.append(clean)
    return frames
