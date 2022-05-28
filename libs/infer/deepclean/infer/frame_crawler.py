import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from deepclean.gwftools.frames import FrameFileFormat, fname_re


def get_prefix(data_dir: Path):
    if not data_dir.exists():
        raise FileNotFoundError("No data directory '{data_dir}'")

    matches = [
        i
        for i in map(fname_re.search, data_dir.iterdir())
        if i is not None and i.group("suffix") == "gwf"
    ]
    if len(matches) == 0:
        raise ValueError(f"No valid .gwf files in data directory '{data_dir}'")

    prefixes = set([i.group("prefix") for i in matches])
    if len(prefixes) > 1:
        raise ValueError(
            "Too many prefixes {} in data directory '{}'".format(
                list(prefixes), data_dir
            )
        )

    lengths = set([i.group("length") for i in matches])
    if len(lengths) > 1:
        raise ValueError(
            "Too many lengths {} in data directory '{}'".format(
                list(lengths), data_dir
            )
        )
    return list(prefixes)[0], int(list(lengths)[0])


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
            tstamps = [int(i.group("t0")) for i in matches if i is not None]
            self.t0 = sorted(tstamps, reverse=self.t0 == -1)[0]

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
            elif (time.time() - start_time) > self.timeout:
                raise RuntimeError(
                    f"No frame file {fname} after {self.timeout}s"
                )

        self.t0 += self.length
        return fname
