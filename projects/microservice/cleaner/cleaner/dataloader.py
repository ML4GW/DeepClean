import time
from pathlib import Path
from queue import Empty, Queue
from typing import Iterable, List

import numpy as np
from gwpy.timeseries import TimeSeries
from microservice.deployment import DataStream
from microservice.frames import load_frame

from deepclean.logging import logger


def witness_iterator(it: Iterable, stride: int) -> np.ndarray:
    try:
        data = next(it)
    except StopIteration:
        raise ValueError("Frame crawler never returned any data")

    idx = 0
    while True:
        start = idx * stride
        stop = (idx + 1) * stride
        if stop > data.shape[-1]:
            try:
                frame = next(it)
            except StopIteration:
                yield np.zeros((stride,), dtype=np.float32), True
                break
            else:
                data = data[:, start:]
                data = np.concatenate([data, frame], axis=-1)
                idx, start, stop = 0, 0, stride

        yield data[:, start:stop], False
        idx += 1


def strain_iterator(q: Queue):
    while True:
        try:
            response = q.get_nowait()
        except Empty:
            time.sleep(1e-3)
        else:
            if response is None:
                break
            yield response


def frame_iter(crawler, channels, sample_rate, q):
    for fname, strain_fname in crawler:
        logger.debug(f"Loading frame files {fname}, {strain_fname}")
        witnesses = load_frame(fname, channels[1:], sample_rate)
        strain = TimeSeries.read(strain_fname, channel=channels[0])
        if strain.sample_rate.value != sample_rate:
            strain = strain.resample(sample_rate)

        q.put((strain, strain_fname))
        yield witnesses
    q.put(None)


def get_data_generators(
    data_dir: Path,
    data_field: str,
    channels: List[str],
    sample_rate: float,
    inference_sampling_rate: float,
    batch_size: int,
    start_first: bool = True,
    timeout: float = 10,
):
    t0 = 0 if start_first else -1
    stream = DataStream(data_dir, data_field)
    crawler = stream.crawl(t0, timeout)
    strain_q = Queue()
    it = frame_iter(crawler, channels, sample_rate, strain_q)

    stride = int(batch_size * sample_rate / inference_sampling_rate)
    witness_it = witness_iterator(it, stride)
    strain_it = strain_iterator(strain_q)
    return witness_it, strain_it
