import logging
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal, Optional, Tuple, Union

import numpy as np
from gwpy.timeseries import TimeSeries, TimeSeriesDict

PATH_LIKE = Union[str, Path]

patterns = {
    "prefix": "[a-zA-Z0-9_:-]+",
    "start": "[0-9]{10}",
    "duration": "[1-9][0-9]*",
    "suffix": "(gwf)|(hdf5)|(h5)",
}
groups = {k: f"(?P<{k}>{v})" for k, v in patterns.items()}
pattern = "{prefix}-{start}-{duration}.{suffix}".format(**groups)
fname_re = re.compile(pattern)


def parse_frame_name(fname: PATH_LIKE) -> Tuple[str, int, int]:
    """Use the name of a frame file to infer its initial timestamp and length

    Expects frame names to follow a standard nomenclature
    where the name of the frame file ends {prefix}_{timestamp}-{length}.gwf

    Args:
        fname: The name of the frame file
    Returns:
        The prefix of the frame file name
        The initial GPS timestamp of the frame file
        The length of the frame file in seconds
    """

    if isinstance(fname, Path):
        fname = fname.name

    match = fname_re.search(fname)
    if match is None:
        raise ValueError(f"Could not parse frame filename {fname}")

    prefix, start, duration, *_ = match.groups()
    return prefix, int(start), int(duration)


@dataclass
class FrameFileFormat:
    prefix: str
    suffix: Literal["gwf", "hdf5", "h5"] = "gwf"

    @classmethod
    def from_frame_file(cls, frame_file: PATH_LIKE):
        prefix, _, __ = parse_frame_name(frame_file)
        suffix = Path(frame_file).suffix[1:]
        return cls(prefix, suffix)

    def get_name(self, timestamp: int, length: int):
        if int(timestamp) != timestamp:
            raise ValueError(f"Timestamp {timestamp} must be an int")
        elif len(str(timestamp)) != 10:
            raise ValueError(
                "Couldn't create valid GPS timestamp from timestamp {}".format(
                    timestamp
                )
            )

        if length <= 0:
            raise ValueError(f"Length {length} must be greater than 0")
        elif int(length) != length:
            raise ValueError(f"Length {length} must be an int")
        elif not 1 <= len(str(length)) < 5:
            raise ValueError(f"Frame length {length} invalid")

        return f"{self.prefix}-{timestamp}-{length}.{self.suffix}"


def _is_gwf(match):
    return match is not None and match.group("suffix") == "gwf"


def get_prefix(data_dir: Path):
    if not data_dir.exists():
        raise FileNotFoundError("No data directory '{data_dir}'")

    fnames = map(str, data_dir.iterdir())
    matches = map(fname_re.search, fnames)
    matches = list(filter(_is_gwf, matches))

    if len(matches) == 0:
        raise ValueError(f"No valid .gwf files in data directory '{data_dir}'")

    prefixes = set([i.group("prefix") for i in matches])
    if len(prefixes) > 1:
        raise ValueError(
            "Too many prefixes {} in data directory '{}'".format(
                list(prefixes), data_dir
            )
        )

    durations = set([i.group("duration") for i in matches])
    if len(durations) > 1:
        raise ValueError(
            "Too many lengths {} in data directory '{}'".format(
                list(durations), data_dir
            )
        )
    return list(prefixes)[0], int(list(durations)[0])


@dataclass
class FrameCrawler:
    data_dir: Path
    t0: Optional[float] = None
    timeout: Optional[float] = None

    def __post_init__(self):
        prefix, self.length = get_prefix(self.data_dir)
        self.file_format = FrameFileFormat(prefix)

        # t0 being None or 0 means start at the first timestamp
        # -1 means start at the last
        if self.t0 is None or self.t0 == 0 or self.t0 == -1:
            matches = map(fname_re.search, map(str, self.data_dir.iterdir()))
            starts = [int(i.group("start")) for i in matches if i is not None]
            self.t0 = sorted(starts, reverse=self.t0 == -1)[0]

    def __iter__(self):
        return self

    def __next__(self):
        fname = self.file_format.get_name(self.t0, self.length)
        fname = self.data_dir / fname

        start_time = time.time()
        i, interval = 1, 3
        while not fname.exists():
            time.sleep(1e-3)
            if (time.time() - start_time) > (i * interval):
                logging.debug(
                    "Waiting for frame files for timestamp {}, "
                    "{}s elapsed".format(self.t0, i * interval)
                )
                i += 1

            if self.timeout is None:
                continue
            elif self.timeout == 0:
                raise StopIteration
            elif (time.time() - start_time) > self.timeout:
                raise RuntimeError(
                    f"No frame file {fname} after {self.timeout}s"
                )

        self.t0 += self.length
        return fname


def load_frame(
    fname: str, channels: Union[str, Iterable[str]], sample_rate: float
) -> np.ndarray:
    """Load the indicated channels from the indicated frame file"""

    if isinstance(channels, str):
        # if we don't have multiple channels, then just grab the data
        data = TimeSeries.read(fname, channels)
        data = data.resample(sample_rate)
        data = data.value
    else:
        # otherwise stack the arrays
        data = TimeSeriesDict.read(fname, channels)
        data = data.resample(sample_rate)
        data = np.stack([data[i].value for i in channels])

    # return as the expected type
    return data.astype("float32")
