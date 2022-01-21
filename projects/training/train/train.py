import os
from typing import List, Optional, Union

from train.data_utils import get_data

from deepclean.logging import configure_logging
from deepclean.trainer.wrapper import make_cmd_line_fn


@make_cmd_line_fn
def main(
    channels: Union[str, List[str]],
    output_directory: str,
    sample_rate: float,
    data_directory: Optional[str] = None,
    t0: Optional[float] = None,
    duration: Optional[float] = None,
    valid_frac: Optional[float] = None,
    force_download: bool = False,
    verbose: bool = False,
    **kwargs
):
    os.makedirs(output_directory, exist_ok=True)
    configure_logging(os.path.join(output_directory, "train.log"), verbose)

    if valid_frac is not None:
        if duration is not None:
            valid_duration = valid_frac * duration
            duration = duration - valid_duration

            if t0 is not None:
                valid_t0 = t0 + duration
            else:
                valid_t0 = None
    else:
        valid_duration = valid_t0 = None

    if data_directory is not None:
        train_fname = os.path.join(data_directory, "training.h5")
        valid_fname = os.path.join(data_directory, "validation.h5")
    else:
        train_fname = valid_fname = None

    train_X, train_y = get_data(
        channels, sample_rate, train_fname, t0, duration, force_download
    )

    if valid_frac is not None or valid_fname is not None:
        try:
            valid_X, valid_y = get_data(
                channels,
                sample_rate,
                valid_fname,
                valid_t0,
                valid_duration,
                force_download,
            )
        except ValueError:
            # if there was a problem getting the validation
            # data caused in some way by the valid_fname
            # not existing, but we didn't explicitly ask for
            # valid_frac, then go ahead and just assume we
            # don't want to do validation and return just
            # the training data
            if valid_frac is None:
                return train_X, train_y

            # otherwise raise whatever the error was
            raise
        return train_X, train_y, valid_X, valid_y
    return train_X, train_y


if __name__ == "__main__":
    main()
