import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator

from gwpy.timeseries import TimeSeries, TimeSeriesDict
from microservice.frames import lock, parse_frame_name

from deepclean.logging import logger


@dataclass
class Writer:
    write_dir: Path
    it: Iterator
    cleaner: Callable
    max_files: int

    def __post_init__(self):
        self.write_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logger.get_logger("DeepClean writer")
        self._files = []

    def __call__(self, **predictions):
        strain, fname = next(self.it)
        timestamp = parse_frame_name(fname.name)[1]
        self.logger.debug(f"Cleaning strain from file {fname}")

        tsd = TimeSeriesDict()
        for key, noise in predictions.items():
            frame = self.cleaner(noise, strain.value)

            model = key.split("-")[1]
            postfix = "DEEPCLEAN-" + model.upper()
            tsd[key] = TimeSeries(
                frame,
                t0=timestamp,
                sample_rate=strain.sample_rate.value,
                channel=strain.channel.name + "-" + postfix,
            )

        write_path = self.write_dir / fname.name
        with lock:
            tsd.write(write_path)

        write_time = time.time()
        tstamp = os.path.getmtime(fname)
        latency = write_time - tstamp

        self._files.append(write_path)
        if len(self._files) > self.max_files:
            fname = self._files.pop(0)
            self.logger.info(f"Removing file {fname} from replay directory")
            fname.unlink()  # TODO: write these somewhere else?

        return write_path, latency
