import logging
from pathlib import Path
from typing import Dict, Optional, Sequence

import h5py
import numpy as np
from gwpy.timeseries import TimeSeries, TimeSeriesDict

from deepclean.gwftools.channels import ChannelList, get_channels
from deepclean.gwftools.frames import fname_re
from deepclean.logging import configure_logging
from deepclean.trainer.wrapper import trainify

FAKE_ID = "FAKE_SINE_FREQ"


def make_fake_sines(
    channels: Sequence[str], t0: float, duration: float, sample_rate: float
) -> Dict[str, np.ndarray]:
    time = np.linspace(t0, t0 + duration, int(duration * sample_rate))

    data = {}
    for channel in channels:
        # TODO: use regex
        f0 = float(channel.split("_")[-1].split("HZ")[0].replace("POINT", "."))
        data[channel] = np.sin(2 * np.pi * f0 * time)
    return data


def fetch(
    channels: Sequence[str],
    t0: float,
    duration: float,
    sample_rate: float,
    nproc: int = 4,
) -> Dict[str, np.ndarray]:
    logging.info(f"Fetching {duration}s of data starting at timestamp {t0}")

    real_channels = [i for i in channels if FAKE_ID not in i]
    fake_channels = [i for i in channels if FAKE_ID in i]

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
                if FAKE_ID in channel:
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
    data: Dict[str, np.ndarray], fname: str, sample_rate: float, t0: float
) -> None:
    """Write to HDF5 format"""

    logging.info(f"Writing data to {fname}")
    with h5py.File(fname, "w") as f:
        for channel, x in data.items():
            dset = f.create_dataset(channel, data=x, compression="gzip")
            dset.attrs["sample_rate"] = sample_rate
            dset.attrs["t0"] = t0
            dset.attrs["channel"] = channel
            dset.attrs["name"] = channel


# note that this function decorator acts both to
# wrap this function such that the outputs of it
# (i.e. the training and possible validation data)
# get passed as inputs to deepclean.trainer.trainer.train,
# as well as to expose these arguments _as well_ as those
# from deepclean.trainer.trainer.train to command line
# execution and parsing
@trainify
def main(
    channels: ChannelList,
    t0: float,
    duration: float,
    sample_rate: float,
    output_directory: Path,
    data_path: Optional[Path] = None,
    valid_frac: Optional[float] = None,
    force_download: bool = False,
    verbose: bool = False,
    **kwargs,
):
    """Train DeepClean on a specified stretch of data

    Run DeepClean's training function on a specified set
    of witness and strain channels from a stretch of data
    specifed by its start time and length, optionally
    reserving some fraction of the data for validation.

    If a path to existing data is specified, an attempt will
    be made to pull the approriate data from that file.
    For each of the specified channels, if the corresponding
    data is either missing, has the incorrect initial timestamp,
    or is insufficiently long, a `ValueError` will be raised.

    Otherwise, the specified data will be fetched from the
    NDS2 server. If a path to data is still specified but
    does not exist, the appropriate directory will be
    created and the downloaded data will be written to a file
    in that directory. If the specified data path doesn't
    adhere to GW file standards, it will be assumed that the
    path is a directory and the data will be written to a
    GW-formatted file in that directory with prefix `"deepclean_train"`.

    Args:
        channels:
            Either a list of channels to use during training,
            or a path to a file containing these channels separated
            by newlines. In either case, it is assumed that the
            0th channel corresponds to the strain data.
        t0:
            Initial GPS timestamp of the stretch of data on which
            to train DeepClean. If `data_path` is specified and
            exists, a `ValueError` will be raised if any of the
            corresponding channels in that file have a `t0` attribute
            that doesn't match this.
        duration:
            Length of the stretch of data on which to train _and validate_
            DeepClean in seconds. If `data_path` is specified and
            exists, each channel in the specified file will be truncated
            to this length, and a `ValueError` will be raised if any of
            them have a duration shorter than this value.
        sample_rate:
            Rate at which to resample witness and strain timeseries.
        data_path:
            Path to existing data stored as an HDF5 file. If this path
            exists and is a file, an attempt will be made to load data
            from this path. If this path exists but is a directory,
            a filename to which the downloaded data will be written
            will be generated in this directory of the form
            `"deepclean_train-{t0}-{duration}.h5"`. Otherwise if the
            path does not exist, whether it refers to a file explicitly
            or a directory to which to form a filename as above will
            be inferred by whether the name of the file adheres to
            GW file naming conventions.
        valid_frac:
            Fraction of training data to reserve for validation, split
            chronologically
        force_download:
            Whether to re-download data even if `data_path` is an
            existing file.
        verbose:
            Indicates whether to log at DEBUG or INFO level

    Returns:
        Array of witness training data timeseries
        Array of strain training data timeseries
        Array of witness validation data timeseries, if validation
            data is requested.
        Array of strain validation data timeseries, if validation
            data is requested.
    """

    output_directory.mkdir(parents=True, exist_ok=True)
    configure_logging(output_directory / "train.log", verbose)
    channels = get_channels(channels)

    if data_path is not None and data_path.is_file() and not force_download:
        data = read(data_path, channels, t0, duration, sample_rate)
    else:
        # in any other situation: no data path, data path isn't
        # an existing file, or always download: always download
        data = fetch(channels, t0, duration, sample_rate)

        # if we specified a data_path, we'll need
        # to write the data we downloaded to it
        if data_path is not None:
            # figure out where to write it to
            match = fname_re.search(data_path.name)
            if match is None or data_path.is_dir():
                # either data_path is an existing directory
                # or data_path's terminal path node isn't formatted
                # like a frame filename, so we'll assume that it
                # refers to a directory and make a default filename
                fname = f"deepclean_train-{t0}-{duration}.h5"
                data_path.mkdir(parents=True, exist_ok=True)
                data_path = data_path / fname
            if match is not None and match.suffix == "gwf":
                # This is formatted like a filename, but the
                # suffix is GWF, and let's not be naming HDF5
                # files like they're GWF files
                raise ValueError("Can't create data file {data_path}")
            elif match is not None:
                # otherwise this refers to a file explicitly, so
                # make sure that the intended directory exists
                data_path.parent.mkdir(parents=True, exist_ok=True)

            write(data, fname, t0, sample_rate)

    X = np.stack([data[chan] for chan in channels[1:]])
    y = data[channels[0]]

    # if we didn't specify to create any validation data,
    # return these arrays as-is
    if valid_frac is None:
        return X, y

    split = int(valid_frac * sample_rate * duration)
    train_X, valid_X = np.split(X, split, axis=1)
    train_y, valid_y = np.split(y, split)
    return train_X, train_y, valid_X, valid_y


if __name__ == "__main__":
    main()
