import sys
import time
import traceback
from multiprocessing import Event, Process, Queue
from pathlib import Path
from queue import Empty
from typing import Optional

import numpy as np
from microservice.frames import FrameCrawler, load_frame

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
    channels = get_channels(channels)
    crawler = FrameCrawler(data_directory, timeout=timeout)
    crawler = iter(crawler)
    start = crawler.t0 + 0

    size = int(duration * sample_rate)
    edge_size = int(cadence * sample_rate)
    X = np.zeros((len(channels), 0))
    y = np.zeros((0,))

    while not event.is_set():
        fname = next(crawler)
        witnesses = load_frame(fname, channels[1:], sample_rate)
        strain = load_frame(fname, channels[0], sample_rate)

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


def target(
    q: Queue,
    event: Event,
    data_directory: Path,
    log_directory: Path,
    channels: ChannelList,
    duration: float,
    cadence: float,
    sample_rate: float,
    timeout: Optional[float] = None,
    verbose: bool = False,
):
    logger.set_logger(
        "DeepClean dataloading",
        log_directory / "dataloader.log",
        verbose=verbose,
    )
    try:
        it = collect_frames(
            event,
            data_directory,
            channels,
            duration,
            cadence,
            sample_rate,
            timeout,
        )
        for X, y, start in it:
            q.put((X, y, start))
    except Exception:
        exc_type, exc, tb = sys.exc_info()
        tb = traceback.format_exception(exc_type, exc, tb)
        tb = "".join(tb[:-1])
        q.put((exc_type, str(exc), tb))
    finally:
        # if we arrived here from an exception, i.e.
        # the event has not been set, then don't
        # close the queue until the parent process
        # has received the error message and set the
        # event itself, otherwise it will never be
        # able to receive the message from the queue
        while not event.is_set():
            time.sleep(1e-3)
        q.close()


def data_collector(
    data_directory: Path,
    log_directory: Path,
    channels: ChannelList,
    duration: float,
    cadence: float,
    sample_rate: float,
    timeout: Optional[float] = None,
    verbose: bool = False,
):
    q = Queue()
    event = Event()
    args = (
        q,
        event,
        data_directory,
        log_directory,
        channels,
        duration,
        cadence,
        sample_rate,
        timeout,
        verbose,
    )

    p = Process(target=target, args=args)
    p.start()
    try:
        while True:
            try:
                X, y, start = q.get_nowait()
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
    finally:
        event.set()
        q.close()
        p.join()
        p.close()
