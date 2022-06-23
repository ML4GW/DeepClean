"""
Running fully asynchronous inference at the moment causes
some inference requests to be dropped on their way back
to the client. To get around this, here are a few tools
to simplify wrapping up the asynchronous processes we would
prefer to use in methods that work synchronously.
"""

import logging
import time
from contextlib import contextmanager
from itertools import chain
from queue import Empty, Queue
from typing import TYPE_CHECKING, Iterable

from hermes.stillwater.utils import ExceptionWrapper

if TYPE_CHECKING:
    from bbhnet.infer.asynchronous import FrameLoader, FrameWriter
    from hermes.stillwater import InferenceClient
    from hermes.stillwater.process import PipelineProcess


class DummyQueue:
    """Extremely dumb queue mocker"""

    def __init__(self):
        self.package = None

    def put(self, package):
        self.package = package

    def get_nowait(self):
        if self.package is None:
            raise Empty

        package = self.package
        self.package = None
        return package


def synchronize(process: "PipelineProcess"):
    """Replace a process's queues with dummys and assign it a logger"""
    process.in_q = DummyQueue()
    process.out_q = DummyQueue()
    process.logger = logging.getLogger(process.name)


@contextmanager
def stream(
    loader: "FrameLoader",
    client: "InferenceClient",
    writer: "FrameWriter",
    inference_sampling_rate: float,
) -> Iterable:
    synchronize(loader)

    def load(fname):
        loader.in_q.put(fname)
        for i in range(int(inference_sampling_rate)):
            yield loader.get_package()

    synchronize(client)
    synchronize(writer)
    writer.in_q = client.out_q

    # we actually do need a legitimate queue for the writer
    writer.out_q = Queue()

    def callback(result, error):
        client.callback(result, error)

        try:
            response = writer.get_package()
            writer.process(response)
        except Exception as e:
            writer.out_q.put(ExceptionWrapper(e))

    def pipeline(crawler, inference_rate):
        frame_it = iter(crawler)
        fname = next(frame_it)
        loader.in_q.put(fname)
        package = loader.get_package()

        while True:
            fname = next(crawler)
            package_it = load(fname)
            if package is None:
                package = next(package_it)

            for package in chain([package], package_it):
                # need to do this whole rigamaroll because
                # client.get_package performs some setting
                # up of inputs and what not that we'll need
                # to actually do the inference. TODO: this
                # logic should be rolled up into a standalone
                # method for this exact reason
                client.in_q.put(package)
                package = client.get_package()
                client.process(*package)
                time.sleep(1 / inference_rate)
            package = None

            # now all the requests are in-flight or being
            # processed in the callback thread, so see if
            # any frames have been written otherwise move on
            try:
                result = writer.out_q.get_nowait()
            except Empty:
                yield None, None
            else:
                # check if what got raised is actually an error
                # and reraise it (with traceback) if so. Otherwise
                # return the written filename and its latency
                if isinstance(result, Exception):  # ExceptionWrapper):
                    raise result  # result.reraise()
                yield result

    with client.client:
        client.client.start_stream(callback=callback)
        yield pipeline
