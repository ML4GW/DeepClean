from pathlib import Path
from typing import List

import h5py
import numpy as np
from gwpy.timeseries import TimeSeries

from deepclean.logging import logger


def validate_csd(
    X: np.ndarray,
    y: np.ndarray,
    channels: List[str],
    sample_rate: float,
    fftlength: float,
    fname: Path,
):
    logger.info(
        "Evaluating CSD of strain channel with witnesses "
        "and saving outputs to {}".format(fname)
    )

    y = TimeSeries(y, sample_rate=sample_rate)
    with h5py.File(fname, "w") as f:
        f.attrs["sample_rate"] = sample_rate
        f.attrs["df"] = 1 / fftlength
        csds = f.create_group("csd")

        for x, channel in zip(X, channels[1:]):
            x = TimeSeries(x, sample_rate=sample_rate)
            csd = y.csd(x, fftlength=fftlength, window="hann")
            csds[channel] = csd.value
