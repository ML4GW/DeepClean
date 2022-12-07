from pathlib import Path
from typing import Optional

import numpy as np

from deepclean.gwftools.channels import ChannelList, get_channels
from deepclean.logging import configure_logging
from deepclean.trainer.wrapper import trainify
from mldatafind.io import read_timeseries


@trainify
def main(
    output_directory: Path,
    data_directory: Path,
    channels: ChannelList,
    t0: float,
    duration: float,
    sample_rate: float,
    valid_frac: Optional[float] = None,
    verbose: bool = False,
    **kwargs,
):
    """Train DeepClean on a specified stretch of data

    Run DeepClean's training function on a specified set
    of witness and strain channels from a stretch of data
    specifed by its start time and length, optionally
    reserving some fraction of the data for validation.

    Args:
        data_directory:
            Directory containing files with the requisite
            stretch of data in HDF5 format.
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
        valid_frac:
            Fraction of training data to reserve for validation, split
            chronologically
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
    tf = t0 + duration

    X = read_timeseries(data_directory, channels, t0, tf, array_like=True)
    if X.shape[-1] != (duration * sample_rate):
        inferred_sample_rate = X.shape[-1] / duration
        raise ValueError(
            "Data found in directory {} contains data "
            "with sample rate {}, expected sample rate {}".format(
                data_directory, inferred_sample_rate, sample_rate
            )
        )
    y, X = np.split(X, [1], axis=0)

    # if we didn't specify to create any validation data,
    # return these arrays as-is
    if valid_frac is None:
        return X, y

    # otherwise carve off the last `valid_frac * duration`
    # seconds worth of data for validation
    split = int((1 - valid_frac) * sample_rate * duration)
    train_X, valid_X = np.split(X, [split], axis=1)
    train_y, valid_y = np.split(y, [split])
    return train_X, train_y, valid_X, valid_y


if __name__ == "__main__":
    main()
