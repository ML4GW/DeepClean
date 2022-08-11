import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from ray import tune

from deepclean.architectures import DeepCleanAE
from deepclean.gwftools.channels import ChannelList, get_channels
from deepclean.gwftools.io import find as find_data
from deepclean.logging import configure_logging
from deepclean.trainer.trainer import train
from deepclean.trainer.wrapper import trainify
from hermes.typeo import typeo


# note that this function decorator acts both to
# wrap this function such that the outputs of it
# (i.e. the training and possible validation data)
# get passed as inputs to deepclean.trainer.trainer.train,
# as well as to expose these arguments _as well_ as those
# from deepclean.trainer.trainer.train to command line
# execution and parsing
@trainify
def get_data(
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
    data = find_data(
        channels,
        t0,
        duration,
        sample_rate,
        data_path=data_path,
        force_download=force_download,
    )

    # create the input and target arrays from the collected data
    X = np.stack([data[channel] for channel in channels[1:]])
    y = data[channels[0]]

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


@typeo
def hp_search(
    output_directory: Path,
    # data params
    channels: ChannelList,
    t0: float,
    duration: float,
    sample_rate: float,
    kernel_length: float,
    kernel_stride: float,
    valid_frac: Optional[float] = None,
    # data loading params
    data_path: Optional[Path] = None,
    force_download: bool = False,
    chunk_length: float = 0,
    num_chunks: int = 1,
    # preproc params
    freq_low: float = 55.0,
    freq_high: float = 65.0,
    filter_order: int = 8,
    # optimization params
    num_trials: int = 10,
    cpus_per_trial: int = 8,
    batch_size: int = 32,
    max_epochs: int = 40,
    init_weights: Optional[Path] = None,
    patience: Optional[int] = None,
    factor: float = 0.1,
    early_stop: int = 20,
    # criterion params
    fftlength: float = 2,
    overlap: Optional[float] = None,
    alpha: float = 1.0,
    # misc params
    profile: bool = False,
    use_amp: bool = False,
    verbose: bool = False,
):
    data = get_data(
        channels=channels,
        t0=t0,
        duration=duration,
        sample_rate=sample_rate,
        output_directory=output_directory,
        data_path=data_path,
        valid_frac=valid_frac,
        force_download=force_download,
        verbose=verbose,
    )
    if valid_frac is not None:
        X, y, valid_X, valid_y = data
        valid_data = (valid_X, valid_y)
    else:
        X, y = data
        valid_data = None

    def trainable(learning_rate, weight_decay):
        history = train(
            architecture=DeepCleanAE,
            output_directory=tune.get_trial_dir(),
            X=X,
            y=y,
            kernel_length=kernel_length,
            kernel_stride=kernel_stride,
            sample_rate=sample_rate,
            chunk_length=chunk_length,
            valid_data=valid_data,
            freq_low=freq_low,
            freq_high=freq_high,
            filter_order=filter_order,
            batch_size=batch_size,
            max_epochs=max_epochs,
            init_weights=init_weights,
            lr=learning_rate,
            weight_decay=weight_decay,
            patience=patience,
            factor=factor,
            early_stop=early_stop,
            fftlength=fftlength,
            overlap=overlap,
            alpha=alpha,
            device="cuda" if torch.cuda.is_available() else "cpu",
            profile=profile,
            use_amp=use_amp,
        )
        return {"score": min(history["valid_loss"])}

    space = {
        "learning_rate": tune.uniform(1e-4, 1e-1),
        "weight_decay": tune.uniform(1e-2),
    }

    output_directory = Path(output_directory)
    results = tune.run(
        trainable,
        config=space,
        num_samples=num_trials,
        mode="min",
        resources_per_trial={"cpu": cpus_per_trial, "gpu": 1},
        local_dir=output_directory.parent,
        name=output_directory.name,
    )
    logging.info("Best ASDR: {}".format(results.best_result["score"]))


if __name__ == "__main__":
    get_data()
