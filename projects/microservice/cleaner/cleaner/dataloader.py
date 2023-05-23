import time
from pathlib import Path
from queue import Empty, Queue
from typing import Iterable, List

import numpy as np
from microservice.deployment import DataStream
from microservice.frames import frame_it


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


def frame_iter(
    crawler: Iterable,
    channels: List[str],
    sample_rate: float,
    inference_sampling_rate: float,
    batch_size: int,
    q: Queue,
):
    stride = int(batch_size * sample_rate / inference_sampling_rate)
    it = frame_it(crawler, channels, sample_rate)

    strain_fname, strain, witnesses = next(it)
    q.put((strain, strain_fname))

    idx = 0
    while True:
        start = idx * stride
        stop = (idx + 1) * stride
        if (idx + 2) * stride > witnesses.shape[-1]:
            try:
                strain_fname, strain, update = next(it)
            except StopIteration:
                yield witnesses[:, start:stop], True
                q.put(None)
                break
            else:
                q.put((strain, strain_fname))
                witnesses = witnesses[:, start:]
                witnesses = np.concatenate([witnesses, update], axis=1)
                idx, start, stop = 0, 0, stride
        yield witnesses[:, start:stop], False


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
    witness_it = frame_iter(
        crawler,
        channels,
        sample_rate,
        inference_sampling_rate,
        batch_size,
        strain_q,
    )
    strain_it = strain_iterator(strain_q)
    return witness_it, strain_it
