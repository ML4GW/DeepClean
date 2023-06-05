from pathlib import Path
from typing import Callable, Dict, Optional

import numpy as np
from microservice.deployment import Deployment, ExportClient
from microservice.frames import get_channels
from trainer.checks import data_checks
from trainer.dataloader import DataCollector

from deepclean.architectures import architectures
from deepclean.logging import logger
from deepclean.trainer.trainer import train
from typeo import scriptify
from typeo.utils import make_dummy


def _intify(x):
    y = int(x)
    y = y if y == x else x
    return str(y)


def _get_str(*times):
    return "-".join(map(_intify, times))


def train_on_segment(
    X: np.ndarray,
    y: np.ndarray,
    deployment: Deployment,
    start: float,
    architecture: Callable,
    sample_rate: float,
    valid_frac: float,
    verbose: bool = False,
    **kwargs,
) -> None:
    duration = len(y) / sample_rate
    str_rep = _get_str(start, start + duration)

    log_file = deployment.log_directory / f"train.{str_rep}.log"

    output_directory = deployment.train_directory / str_rep
    output_directory.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving training outputs to directory {output_directory}")

    # now set the logger to one specific to this training run
    logger.set_logger(f"DeepClean train {str_rep}", log_file, verbose)

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


exclude = ["X", "y", "architecture", "valid_data", "output_directory"]


@scriptify(
    kwargs=make_dummy(train, exclude=exclude),
    architecture=architectures,
)
def main(
    run_directory: Path,
    data_directory: Path,
    ifo: str,
    export_endpoint: str,
    channels: Dict[str, str],
    architecture: Callable,
    train_duration: float,
    retrain_cadence: float,
    sample_rate: float,
    valid_frac: float,
    fine_tune_decay: Optional[float] = None,
    verbose: bool = False,
    **kwargs,
) -> None:
    # build a deployment object that contains all
    # the structure of where our logs/intermediate
    # data products go
    deployment = Deployment(run_directory)
    log_file = deployment.log_directory / "train.root.log"
    root_logger = logger.set_logger(
        "DeepClean trainer", log_file, verbose=verbose
    )

    # connect to the export service
    export_client = ExportClient(export_endpoint)

    # start collecting frames from the data stream
    # to aggregate into a dataset
    channels = get_channels(channels[ifo])
    frame_collector = DataCollector(
        data_directory,
        ifo,
        deployment.log_directory,
        channels,
        train_duration,
        retrain_cadence,
        sample_rate,
        timeout=3,
        verbose=verbose,
    )

    # enter it as a context since it does the
    # collection in a background process while
    # we train in the main process
    last_start = None
    with frame_collector as data_it:
        for X, y, start in data_it:
            span = _get_str(start, start + train_duration)

            # perform some checks on the training dataset
            # to see if it's even worth attemptint to train
            # a model on it
            fname = deployment.data_checks_directory / f"{span}.h5"
            passed = data_checks(
                X, y, channels, sample_rate, fftlength=8, fname=fname
            )
            if not passed:
                logger.warning(
                    "Training dataset {} did not pass quality checks, "
                    "skipping training on this dataset"
                )
                continue

            # check to see if this dataset falls directly
            # after the previous one, or if we need to
            # take some special steps to reset after a
            # dropped dataset
            if last_start is not None:
                expected_start = last_start + retrain_cadence
                if start > expected_start:
                    # TODO: insert any logic about how we do
                    # training differently on a new lock segment
                    pass

            # launch a training job on the current dataset
            root_logger.info(f"Launching training on segment {span}")
            weights_path = train_on_segment(
                X,
                y,
                deployment=deployment,
                start=start,
                architecture=architecture,
                sample_rate=sample_rate,
                valid_frac=valid_frac,
                verbose=verbose,
                **kwargs,
            )
            logger.set_logger("DeepClean train")
            logger.info(
                "Training on segment {} complete, saved "
                "optimized weights to {}".format(span, weights_path)
            )

            # send the optimized weights to the export service
            # to add the trained model to our model repository
            export_client.export(weights_path)

            # for trainings after the first, use the previous
            # optimized weights and reduce the learning rate
            kwargs["init_weights"] = weights_path
            if last_start is None:
                kwargs["lr"] = kwargs["lr"] * fine_tune_decay
            last_start = start
