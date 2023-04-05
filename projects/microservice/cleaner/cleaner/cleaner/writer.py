from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator

from gwpy.timeseries import TimeSeries, TimeSeriesDict
from microservice.frames import parse_frame_name


@dataclass
class Writer:
    write_dir: Path
    it: Iterator
    cleaner: Callable
    sample_rate: float

    def __post_init__(self):
        self.write_dir.mkdir(parents=True, exist_ok=True)

    def __call__(self, **predictions):
        strain, fname = next(self.it)
        timestamp = parse_frame_name(fname.name)[1]

        tsd = TimeSeriesDict()
        for key, noise in predictions.items():
            frame = self.cleaner(noise, strain.value)

            model = key.split("-")[1]
            postfix = "DEEPCLEAN-" + model.upper()
            tsd[key] = TimeSeries(
                frame,
                t0=timestamp,
                sample_rate=self.sample_rate,
                channel=strain.channel + "-" + postfix
            )
        tsd.write(self.write_dir / fname)
        return self.write_dir / fname