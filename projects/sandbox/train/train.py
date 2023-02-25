from pathlib import Path
from typing import Callable, List, Optional

import numpy as np

from deepclean.architectures import architectures
from deepclean.gwftools.channels import ChannelList, get_channels
from deepclean.logging import logger
from deepclean.trainer import train
from mldatafind.io import read_timeseries
from typeo import scriptify
from typeo.utils import make_dummy


def _intify(x):
    y = int(x)
    y = y if y == x else x
    return str(y)


def _get_str(*times):
    return "-".join(map(_intify, times))


def read_segments(segment_file: Path):
    segments = [i for i in segment_file.read_text().splitlines() if i]
    segments = [tuple(map(float, i.split(","))) for i in segments]
    return segments


def train_on_segment(
    output_directory: Path,
    log_file: Path,
    data_directory: Path,
    start: float,
    stop: float,
    channels: List[str],
    architecture: Callable,
    sample_rate: float,
    valid_frac: float,
    verbose: bool = False,
    **kwargs,
) -> None:
    output_directory.mkdir(parents=True, exist_ok=True)
    str_rep = _get_str(start, stop)
    logger.set_logger(f"DeepClean train {str_rep}", log_file, verbose)

    duration = stop - start
    X, _ = read_timeseries(
        data_directory, channels, start, stop, array_like=True
    )

    if X.shape[-1] != (duration * sample_rate):
        inferred_sample_rate = X.shape[-1] / duration
        raise ValueError(
            "Data found in directory {} contains data "
            "with sample rate {}, expected sample rate {}".format(
                data_directory, inferred_sample_rate, sample_rate
            )
        )
    [y], X = np.split(X, [1], axis=0)

    # carve off the last `valid_frac * duration`
    # seconds worth of data for validation
    split = int((1 - valid_frac) * sample_rate * duration)
    train_X, valid_X = np.split(X, [split], axis=1)
    train_y, valid_y = np.split(y, [split])
    kwargs["valid_data"] = (valid_X, valid_y)

    return train(
        X=train_X,
        y=train_y,
        output_directory=output_directory,
        architecture=architecture,
        sample_rate=sample_rate,
        **kwargs,
    )


@scriptify(
    kwargs=make_dummy(train, exclude=["X", "y", "architecture"]),
    architecture=architectures,
)
def main(
    output_directory: Path,
    data_directory: Path,
    segment_file: Path,
    channels: ChannelList,
    architecture: Callable,
    train_duration: float,
    retrain_cadence: float,
    sample_rate: float,
    valid_frac: float,
    fine_tune_decay: Optional[float] = None,
    min_test_required: float = 1024,
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
    log_directory = output_directory / "log"
    log_directory.mkdir(parents=True, exist_ok=True)
    root_logger = logger.set_logger(
        "DeepClean", log_directory / "train.log", verbose=verbose
    )

    channels = get_channels(channels)
    segments = read_segments(segment_file)

    run_requires = train_duration - min_test_required
    results_dir = None
    lr = kwargs["lr"] * 1
    for start, stop in segments:
        duration = stop - start
        num_trains = int((duration - run_requires) // retrain_cadence) + 1
        root_logger.info(
            "Running {} trainings on segment {}".format(
                num_trains, _get_str(start, stop)
            )
        )

        for i in range(num_trains):
            if fine_tune_decay is not None and results_dir is not None:
                kwargs["init_weights"] = results_dir
                kwargs["lr"] = lr * fine_tune_decay

            train_start = start + i * retrain_cadence
            train_stop = train_start + train_duration
            str_rep = _get_str(train_start, train_stop)

            results_dir = output_directory / f"train_{str_rep}"
            log_file = log_directory / f"train_{str_rep}.log"
            root_logger.info(f"Beginning training on segment {str_rep}")
            train_on_segment(
                results_dir,
                log_file,
                data_directory,
                architecture=architecture,
                start=train_start,
                stop=train_stop,
                channels=channels,
                sample_rate=sample_rate,
                valid_frac=valid_frac,
                verbose=verbose,
                **kwargs,
            )


if __name__ == "__main__":
    main()
