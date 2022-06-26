import logging
import re
from pathlib import Path
from typing import Dict, Optional, Sequence

import h5py
import numpy as np
from gwpy.timeseries import TimeSeries, TimeSeriesDict

from deepclean.gwftools.channels import ChannelList
from deepclean.gwftools.frames import fname_re

FAKE_REGEX = re.compile(r"FAKE_SINE_FREQ_(?P<freq>\d+POINT\d+)HZ")


def make_fake_sines(
    channels: Sequence[str], t0: float, duration: float, sample_rate: float
) -> Dict[str, np.ndarray]:
    time = np.arange(t0, t0 + duration, 1 / sample_rate)

    data = {}
    for channel in channels:
        freq = FAKE_REGEX.search(channel)
        if freq is None:
            raise ValueError(
                f"Fake sine channel '{channel}' not properly formatted"
            )

        freq = freq.group("freq")
        freq = float(freq.replace("POINT", "."))
        data[channel] = np.sin(2 * np.pi * freq * time)
    return data


def fetch(
    channels: Sequence[str],
    t0: float,
    duration: float,
    sample_rate: float,
    nproc: int = 4,
) -> Dict[str, np.ndarray]:
    logging.info(f"Fetching {duration}s of data starting at timestamp {t0}")

    fake_channels = list(filter(FAKE_REGEX.fullmatch, channels))
    real_channels = [i for i in channels if i not in fake_channels]

    # get data and resample
    # TODO: do any logic about data quality?
    data = TimeSeriesDict.get(
        real_channels, t0, t0 + duration, nproc=nproc, allow_tape=True
    )
    data = data.resample(sample_rate)
    data = {channel: x.value for channel, x in data.items()}

    # make any sinusoids
    fake_data = make_fake_sines(fake_channels, t0, duration, sample_rate)
    data.update(fake_data)

    return data


def read(
    fname: Path,
    channels: Sequence[str],
    t0: float,
    duration: float,
    sample_rate: float,
) -> Dict[str, np.ndarray]:
    logging.info(f"Reading data from '{fname}'")

    data = {}
    with h5py.File(str(fname), "r") as f:
        for channel in channels:
            try:
                dataset = f[channel]
            except KeyError:
                if FAKE_REGEX.fullmatch(channel) is not None:
                    # if the channel is missing because it's a
                    # fake sinusoid, generate it now
                    fake_data = make_fake_sines(
                        [channel], t0, duration, sample_rate
                    )
                    data.update(fake_data)
                    continue
                else:
                    # otherwise vocalize our unhappiness
                    raise ValueError(
                        "Data file {} doesn't contain channel {}".format(
                            fname, channel
                        )
                    )

            # enforce that the initial timestamps match,
            # otherwise we're just talking about different data
            if dataset.attrs["t0"] != t0:
                raise ValueError(
                    "Channel {} has initial GPS timestamp {}, "
                    "expected {}".format(channel, dataset.attrs["t0"], t0)
                )

            # now grab the actual data array and preprocess it
            x = dataset[:]

            # resample if necessary TODO: raise an error if this
            # resampling is an upsample?
            if sample_rate != dataset.attrs["sample_rate"]:
                x = TimeSeries(x, dt=1 / dataset.attrs["sample_rate"])
                x = x.resample(sample_rate).value

            # allow for shorter durations so that we only need
            # to download one stretch of long data then reuse
            # it for shorter stretches if need be
            if len(x) < (sample_rate * duration):
                raise ValueError(
                    "Channel {} has only {}s worth of data, "
                    "expected at least {}".format(
                        channel, len(x) // sample_rate, duration
                    )
                )

            # grab the first `duration` seconds of data
            data[channel] = x[: int(sample_rate * duration)]
    return data


def write(
    data: Dict[str, np.ndarray], fname: Path, t0: float, sample_rate: float
) -> None:
    """Write to HDF5 format"""

    logging.info(f"Writing data to {fname}")
    with h5py.File(str(fname), "w") as f:
        for channel, x in data.items():
            # don't write fake sinusoids
            if FAKE_REGEX.fullmatch(channel) is not None:
                continue

            dset = f.create_dataset(channel, data=x, compression="gzip")
            dset.attrs["sample_rate"] = sample_rate
            dset.attrs["t0"] = t0


def find(
    channels: ChannelList,
    t0: float,
    duration: float,
    sample_rate: float,
    data_path: Optional[Path] = None,
    force_download: bool = False,
):
    if data_path is None:
        # we didn't specify a path at all,
        # so don't bother reading or writing the data
        # and just download it into memory
        data = fetch(channels, t0, duration, sample_rate)
    else:
        # we specified a data path of some kind, figure
        # out if it refers to a file or a directory
        match = fname_re.search(data_path.name)

        if data_path.is_file() or match is not None:
            # either the path refers to an existing file,
            # or to a path whose name is formatted like a file
            if not data_path.is_file() and match.group("suffix") != "gwf":
                # the file doesn't exist yet, but it's a valid
                # filename because it doesn't end in .gwf. Make
                # the path to the directory containing this file
                # if it doesn't exist yet
                data_path.parent.mkdir(parents=True, exist_ok=True)
            elif not data_path.is_file():
                # the file dosn't exist yet, but it's formatted like
                # a GWF file and we're dealing with HDF5s, so let's
                # raise an error to keep from muddying the waters
                raise ValueError(f"Can't create data file {data_path}")
        elif match is None or data_path.is_dir():
            # either data_path is an existing directory
            # or data_path's terminal path node isn't formatted
            # like a frame filename, so we'll assume that it
            # refers to a directory and make a default filename
            fname = f"deepclean_train-{t0}-{duration}.h5"
            data_path.mkdir(parents=True, exist_ok=True)
            data_path = data_path / fname

        if data_path.is_file() and not force_download:
            # if the data exists and we're not forcing a fresh
            # download, attempt to read the existing data
            data = read(data_path, channels, t0, duration, sample_rate)
        else:
            # otherwise, download the data and
            # write it to the specified path
            data = fetch(channels, t0, duration, sample_rate)
            write(data, data_path, t0, sample_rate)
    return data
