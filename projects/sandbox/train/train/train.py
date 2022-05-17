import os
from typing import Optional

from train.data_utils import get_data

from deepclean.gwftools.channels import ChannelList, get_channels
from deepclean.logging import configure_logging
from deepclean.trainer.wrapper import trainify


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
    """Train DeepClean on archival NDS2 data

    Args:
        channels:
            Either a list of channels to use during training,
            or a path to a file containing these channels separated
            by newlines. In either case, it is assumed that the
            0th channel corresponds to the strain data.
        output_directory:
            Location to which to save training logs and outputs
        sample_rate:
            Rate at which to resample witness and strain timeseries
        data_directory:
            Path to existing data stored as HDF5 files. These files
            are assumed to have the names "training.h5" and "validation.h5",
            and to contain the channel data at the root level. If these
            files do not exist and `t0` and `duration` are specified, data
            will be downloaded and saved to these files. If left as `None`,
            data will be downloaded but not saved.
        t0:
            Initial GPS timestamp of the stretch of data on which
            to train DeepClean. Only necessary if `data_directory`
            isn't specified or the files indicated above don't exist,
            or if `force_download == True`.
        duration:
            Length of the stretch of data on which to train _and validate_
            DeepClean in seconds. Only necessary if `data_directory`
            isn't specified or the files indicated above don't exist,
            or if `force_download == True`.
            If still specified anyway, the training data in
            `{data_directory}/training.h5` will be truncated to this
            length, discarding the _earliest_ data.
        valid_frac:
            Fraction of training data to reserve for validation, split
            chronologically. If not `None` and data needs to be
            downloaded, the first `(1 - valid_frac) * duration` seconds of
            data will be used for training, and the last
            `valid_frac * duration` seconds will be used for validation.
            If `None` and data needs to be downloaded, none will be
            reserved for validation.
        force_download:
            Whether to re-download data even if `data_directory` is not
            `None` and `"training.h5"` and `"validation.h5"` exist there.
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

    os.makedirs(output_directory, exist_ok=True)
    configure_logging(os.path.join(output_directory, "train.log"), verbose)

    # decide whether we have enough info to download
    # validation data if we need to
    valid_t0 = valid_duration = None
    if valid_frac is not None and duration is not None:
        # we indicated a valid fraction and a duration,
        # so break up the duration into training and
        # validation segments
        valid_duration = valid_frac * duration
        duration = duration - valid_duration

        # we specified an initial timestamp, so use
        # this plus the reduced `duration` to calculate
        # the initial timestamp of the validation data
        if t0 is not None:
            valid_t0 = t0 + duration

    # if we specified a data directory, build the
    # assumed filenames we're looking for in it
    if data_directory is not None:
        train_fname = os.path.join(data_directory, "training.h5")
        valid_fname = os.path.join(data_directory, "validation.h5")
    else:
        train_fname = valid_fname = None

    # read channels from text file if one was specified
    channels = get_channels(channels)

    # grab training arrays either from local data
    # or from NDS2 server if it doesn't exist locally
    train_X, train_y = get_data(
        channels, sample_rate, train_fname, t0, duration, force_download
    )

    # try to get validation data if any was requested
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

    # otherwise just return the training data
    return train_X, train_y


if __name__ == "__main__":
    main()
