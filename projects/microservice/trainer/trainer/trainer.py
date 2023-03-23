from pathlib import Path
from typing import Callable, Optional

import numpy as np
from trainer.dataloader import DataCollector

from deepclean.architectures import architectures
from deepclean.logging import logger
from deepclean.trainer.trainer import train
from deepclean.utils.channels import ChannelList
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
    output_directory: Path,
    start: float,
    architecture: Callable,
    sample_rate: float,
    valid_frac: float,
    verbose: bool = False,
    **kwargs,
) -> None:
    duration = len(y) / sample_rate
    str_rep = _get_str(start, start + duration)

    log_file = output_directory / "log" / f"train_{str_rep}.log"

    output_directory = output_directory / "training" / str_rep
    logger.info(f"Saving training outputs to directory {output_directory}")
    output_directory.mkdir(parents=True, exist_ok=True)

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


@scriptify(
    kwargs=make_dummy(train, exclude=["X", "y", "architecture", "valid_data"]),
    architecture=architectures,
)
def main(
    output_directory: Path,
    data_directory: Path,
    channels: ChannelList,
    architecture: Callable,
    train_duration: float,
    retrain_cadence: float,
    sample_rate: float,
    valid_frac: float,
    fine_tune_decay: Optional[float] = None,
    verbose: bool = False,
    **kwargs,
):
    log_directory = output_directory / "log"
    log_directory.mkdir(parents=True, exist_ok=True)
    root_logger = logger.set_logger(
        "DeepClean trainer", log_directory / "root.log", verbose=verbose
    )

    frame_collector = DataCollector(
        data_directory,
        log_directory,
        channels,
        train_duration,
        retrain_cadence,
        sample_rate,
        timeout=60,
        verbose=verbose,
    )

    last_start = None
    with frame_collector as data_it:
        for X, y, start in data_it:
            if last_start is not None:
                expected_start = last_start + retrain_cadence
                if start > expected_start:
                    # TODO: insert any logic about how we do
                    # training differently on a new lock segment
                    pass

            root_logger.info(
                "Launching training on segment {}-{}".format(
                    start, start + train_duration
                )
            )

            weights_path = train_on_segment(
                X,
                y,
                output_directory=output_directory,
                start=start,
                architecture=architecture,
                sample_rate=sample_rate,
                valid_frac=valid_frac,
                verbose=verbose,
                **kwargs,
            )

            logger.set_logger("DeepClean train")
            logger.info(
                "Training on segment {}-{} complete, saved "
                "optimized weights to {}".format(
                    start, start + train_duration, weights_path
                )
            )

            # for trainings after the first, use the previous
            # optimized weights and reduce the learning rate
            kwargs["init_weights"] = weights_path
            if last_start is None:
                kwargs["lr"] = kwargs["lr"] * fine_tune_decay
            last_start = start
