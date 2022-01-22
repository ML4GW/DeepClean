import logging
import os
from typing import Dict, List, Optional

import h5py
import numpy as np
from gwpy.timeseries import TimeSeriesDict


def fetch(
    channels: List[str],
    t0: float,
    duration: float,
    sample_rate: float,
    nproc: int = 4,
) -> Dict[str, np.ndarray]:
    logging.info(f"Fetching {duration}s of data starting at timestamp {t0}")

    fake_id = "FAKE_SINE_FREQ"
    real_channels = [i for i in channels if fake_id not in i]
    fake_channels = [i for i in channels if fake_id in i]

    # get data and resample
    data = TimeSeriesDict.get(
        real_channels, t0, t0 + duration, nproc=nproc, allow_tape=True
    )
    data = data.resample(sample_rate)
    data = {channel: x.value for channel, x in data.items()}

    time = np.linspace(t0, t0 + duration, int(duration * sample_rate))
    for channel in fake_channels:
        # TODO: use regex
        f0 = float(channel.split("_")[-1].split("HZ")[0].replace("POINT", "."))
        data[channel] = np.sin(2 * np.pi * f0 * time)
    return data


def read(
    fname: str, channels: List[str], sample_rate: Optional[float]
) -> Dict[str, np.ndarray]:
    logging.info(f"Reading data from '{fname}'")
    data = {}
    with h5py.File(fname, "r") as f:
        for chan, x in f.items():
            if chan not in channels:
                continue
            data[chan] = x[:]

            if sample_rate is None:
                sample_rate = x.attrs["sample_rate"]
            elif sample_rate != x.attrs["sample_rate"]:
                raise ValueError(
                    "Channel {} has sample rate {}Hz, expected "
                    "sample rate of {}Hz".format(
                        chan, data.attrs["sample_rate"], sample_rate
                    )
                )
    return data


def write(
    data: Dict[str, np.ndarray], fname: str, sample_rate: float, t0: float
):
    """Write to HDF5 format"""

    logging.info(f"Writing data to {fname}")
    with h5py.File(fname, "w") as f:
        for channel, x in data.items():
            dset = f.create_dataset(channel, data=x, compression="gzip")
            dset.attrs["sample_rate"] = sample_rate
            dset.attrs["t0"] = t0
            dset.attrs["channel"] = channel
            dset.attrs["name"] = channel


def get_data(
    channels: List[str],
    sample_rate: Optional[float] = None,
    fname: Optional[str] = None,
    t0: Optional[str] = None,
    duration: Optional[str] = None,
    force_download: bool = False,
):
    cant_download = any([i is None for i in (t0, duration, sample_rate)])
    if fname is None and cant_download:
        raise ValueError(
            "Must specify either a data directory or "
            "all of a start time, duration, and sample rate"
        )
    elif cant_download:
        # we specified a filename, but we're missing
        # either a start time or a duration
        if not os.path.exists(fname):
            # the filename doesn't exist, in which case
            # we don't know what data to download, so
            # raise an error
            raise ValueError(
                "Data file {} does not exist and "
                "one of either a start time, duration "
                "or sampling rate weren't specified".format(fname)
            )
        elif force_download:
            # we indicated that we wanted to download
            # the data even though it already exists,
            # but we don't know what data to download
            raise ValueError(
                "Can't force download without all of "
                "start time, duration, and sampling rate specified"
            )

        # read from the indicated file
        data = read(fname, channels, sample_rate)
    elif fname is not None:
        # we indicated a file, but also a t0 and duration
        if not os.path.exists(fname) or force_download:
            # use the t0 and duration if the file doesn't
            # exist or if we indicated that we'd like
            # to re-download the data again anyway. In this
            # case be sure to write the data to the indicated file
            data = fetch(channels, t0, duration, sample_rate)

            os.makedirs(os.path.dirname(fname), exist_ok=True)
            write(data, fname, sample_rate, t0)
        else:
            # otherwise read the existing data
            data = read(fname, channels, sample_rate)
    else:
        # we didn't specify a file, but both a t0 and duration,
        # so fetch that stretch of data from nds
        data = fetch(channels, t0, duration, sample_rate)

    X = np.stack([data[chan] for chan in sorted(channels[1:])])
    y = data[channels[0]]
    return X, y
