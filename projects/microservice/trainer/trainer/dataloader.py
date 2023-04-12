import sys
import time
import traceback
from dataclasses import dataclass
from multiprocessing import Event, Process, Queue
from pathlib import Path
from queue import Empty
from typing import Optional

import numpy as np
from microservice.deployment import DataStream
from microservice.frames import load_frame

from deepclean.logging import logger
from deepclean.utils.channels import ChannelList, get_channels


def collect_frames(
    event: Event,
    data_directory: Path,
    channels: ChannelList,
    duration: float,
    cadence: float,
    sample_rate: float,
    timeout: Optional[float] = None,
) -> np.ndarray:
    stream = DataStream(data_directory)
    crawler = stream.crawl(t0=0, timeout=timeout)

    channels = get_channels(channels)
    start = None

    size = int(duration * sample_rate)
    edge_size = int(cadence * sample_rate)
    X = np.zeros((len(channels) - 1, 0))
    y = np.zeros((0,))

    # clear the first couple filenames to avoid
    # a race condition where the first file
    # accidentally gets cycled out before we start
    for _ in range(2):
        next(crawler)

    while not event.is_set():
        fname, strain_fname = next(crawler)
        if start is None:
            start = int(fname.stem.split("-")[-2])

        witnesses = load_frame(fname, channels[1:], sample_rate)
        strain = load_frame(strain_fname, channels[0], sample_rate)

        X = np.concatenate([X, witnesses], axis=1)
        y = np.concatenate([y, strain])

        # TODO: insert some checks here about data quality,
        # checking for coherence, etc., and possibly reset
        # X, y, and start accordingly once the conditions have
        # been met again
        if False:
            logger.info(
                "Data quality conditions have been violated, "
                "pausing data collection until conditions "
                "improve."
            )

        if len(y) >= size:
            logger.info(
                "Accumulated {}s of data starting at "
                "GPS time {}, passing to training".format(duration, start)
            )
            yield X, y, start

            X = X[:, edge_size:]
            y = y[edge_size:]
            start += cadence


@dataclass
class DataCollector:
    data_directory: Path
    log_directory: Path
    channels: ChannelList
    duration: float
    cadence: float
    sample_rate: float
    timeout: Optional[float] = None
    verbose: bool = False

    def __enter__(self):
        self.q = Queue()
        self.event = Event()
        self.done_event = Event()
        self.clear_event = Event()
        self.p = Process(target=self)
        self.p.start()
        return self._iter_through_q()

    def __exit__(self, *_):
        # set the event to let the child process
        # know that we're done with whatever data
        # it's generating and it should stop
        self.event.set()

        # wait for the child to indicate to us
        # that it has been able to finish gracefully
        while not self.done_event.is_set():
            time.sleep(1e-3)

        # remove any remaining data from the queue
        # to flush the child process's buffer so
        # that it can exit gracefully, then close
        # the queue from our end
        self._clear_q()
        self.q.close()
        self.clear_event.set()

        # now wait for the child to exit
        # gracefully then close it
        self.p.join()
        self.p.close()

    def _iter_through_q(self):
        while True:
            try:
                X, y, start = self.q.get_nowait()
            except Empty:
                time.sleep(1)
                continue

            if isinstance(start, str):
                exc_type, msg, tb = X, y, start
                logger.exception(
                    "Encountered exception in data collection process:\n" + tb
                )
                raise exc_type(msg)
            yield X, y, start

    def _clear_q(self):
        """Greedily empty our internal queue"""

        while True:
            try:
                self.q.get_nowait()
            except Empty:
                break
            time.sleep(1e-3)

    def __call__(self):
        logger.set_logger(
            "DeepClean dataloading",
            self.log_directory / "dataloader.log",
            verbose=self.verbose,
        )
        try:
            it = collect_frames(
                self.event,
                self.data_directory,
                self.channels,
                self.duration,
                self.cadence,
                self.sample_rate,
                self.timeout,
            )
            for X, y, start in it:
                self.q.put((X, y, start))
        except Exception:
            exc_type, exc, tb = sys.exc_info()
            tb = traceback.format_exception(exc_type, exc, tb)
            tb = "".join(tb[:-1])
            self.q.put((exc_type, str(exc), tb))
        finally:
            # now let the parent process know that
            # there's no more information going into
            # the queue, and it's free to empty it
            self.done_event.set()

            # if we arrived here from an exception, i.e.
            # the event has not been set, then don't
            # close the queue until the parent process
            # has received the error message and set the
            # event itself, otherwise it will never be
            # able to receive the message from the queue
            while not self.event.is_set() or not self.clear_event.is_set():
                time.sleep(1e-3)

            self.q.close()
            self.q.cancel_join_thread()
