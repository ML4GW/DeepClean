import time
from pathlib import Path
from queue import Empty, Queue
from typing import Iterable, List

import numpy as np
from microservice.frames import FrameCrawler, load_frame

from deepclean.logging import logger


def witness_iterator(it: Iterable, stride: int) -> np.ndarray:
    try:
        data = next(it)
    except StopIteration:
        raise ValueError("Frame crawler never returned any data")

    sequence_end, idx = False, 0
    while not sequence_end:
        start = idx * stride
        stop = (idx + 1) * stride
        if (idx + 2) * stride > data.shape[-1]:
            try:
                frame = next(it)
            except StopIteration:
                sequence_end = True
            else:
                data = data[:, start:]
                data = np.concatenate([data, frame], axis=-1)
                idx, start, stop = 0, 0, stride
        yield data[:, start: stop], sequence_end


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
    for fname in crawler:
        logger.debug(f"Loading frame file {fname}")
        frame = load_frame(fname, channels, sample_rate)
        strain, witnesses = np.split(frame, [1], axis=0)
        q.put((strain, fname))
        yield witnesses
    q.put(None)


def get_data_generators(
    witness_data_dir: Path,
    channels: List[str],
    sample_rate: float,
    inference_sampling_rate: float,
    batch_size: int,
    start_first: bool = True,
    timeout: float = 10
):
    t0 = 0 if start_first else -1
    crawler = FrameCrawler(witness_data_dir, t0, timeout)
    strain_q = Queue()
    it = frame_iter(crawler, channels, sample_rate, strain_q)

    stride = int(batch_size * sample_rate / inference_sampling_rate)
    witness_it = witness_iterator(it, stride)
    strain_it = strain_iterator(strain_q)
    return witness_it, strain_it
