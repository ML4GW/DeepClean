import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from deepclean.gwftools import FrameFileFormat, fname_re


def get_prefix(data_dir: Path):
    if not os.path.exists(data_dir):
        raise FileNotFoundError("No data directory '{data_dir}'")

    fnames = os.listdir(data_dir)
    matches = [i for i in map(fname_re.search, fnames) if i is not None]
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
    return prefixes[0], int(lengths[0])


@dataclass
class FrameCrawler:
    witness_data_dir: Path
    strain_data_dir: Path
    start_first: bool = False
    timeout: Optional[float] = None

    def __post_init__(self):
        witness_prefix, self.length = get_prefix(self.witness_data_dir)
        strain_prefix, length = get_prefix(self.strain_data_dir)

        if length != self.length:
            raise ValueError(
                "Lengths of frames in witness data directory {} "
                "don't match length of frames in strain data "
                "directory {}".format(self.length, length)
            )

        self.witness_format = FrameFileFormat(witness_prefix)
        self.strain_format = FrameFileFormat(strain_prefix)

        self.t0 = None

    def __iter__(self):
        witness_fnames = os.listdir(self.witness_data_dir)
        matches = map(fname_re.search, witness_fnames)
        tstamps = [i.group("t0") for i in matches if i is not None]

        self.t0 = sorted(tstamps, reverse=not self.start_first)[0]
        return self

    def __next__(self):
        if self.t0 is None:
            self.__iter__()

        witness_fname = self.witness_format.get_name(self.t0, self.length)
        strain_fname = self.strain_format.get_name(self.t0, self.length)

        witness_fname = self.witness_data_dir / witness_fname
        strain_fname = self.strain_data_dir / strain_fname

        start_time = time.time()
        i = 1
        while not (
            os.path.exists(witness_fname) and os.path.exists(strain_fname)
        ):
            time.sleep(1e-3)
            if (
                self.timeout is not None
                and (time.time() - start_time) > self.timeout
            ):
                raise RuntimeError(
                    "No witness or strain file for timestamp {} "
                    "in directories {} or {} after {}s".format(
                        self.t0,
                        self.witness_data_dir,
                        self.strain_data_dir,
                        self.timeout,
                    )
                )

            if (time.time() - start_time) > (i * 10):
                logging.debug(
                    "Waiting for frame files for timestamp {}, "
                    "{}s elapsed".format(self.t0, i * 10)
                )
                i += 1

        self.t0 += self.length
        return witness_fname, strain_fname
