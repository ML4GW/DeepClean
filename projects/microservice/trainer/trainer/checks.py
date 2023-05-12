from pathlib import Path
from typing import List

import h5py
import numpy as np
from gwpy.timeseries import TimeSeries

from deepclean.logging import logger


def check_coherences(
    X: np.ndarray,
    y: np.ndarray,
    channels: List[str],
    fftlength: float,
    sample_rate: float,
):
    """
    Measure the coherence of each witness channel
    with the strain channel. TODO: add some checks
    around having some minimal coherence in the
    target frequency range.
    """
    logger.info("Evaluating coherece of strain channel with witnesses")
    psd_kwargs = dict(fftlength=fftlength, window="hann", method="median")
    y = TimeSeries(y, sample_rate=sample_rate)
    pyy = y.psd(**psd_kwargs)

    coherences = {}
    for x, channel in zip(X, channels[1:]):
        x = TimeSeries(x, sample_rate=sample_rate)
        pxx = x.psd(**psd_kwargs)
        pxy = y.csd(x, **psd_kwargs)
        coh = pxy**2 / (pxx * pyy)
        coherences[channel] = coh.value**0.5
    return coherences


def data_checks(
    X: np.ndarray,
    y: np.ndarray,
    channels: List[str],
    sample_rate: float,
    fftlength: float,
    fname: Path,
) -> bool:
    logger.info(
        "Performing checks on training data and "
        "saving outputs to {}".format(fname)
    )
    with h5py.File(fname, "w") as f:
        f.attrs["sample_rate"] = sample_rate
        f.attrs["df"] = 1 / fftlength

        group = f.create_group("coherence")
        coherences = check_coherences(X, y, channels, sample_rate, fftlength)
        group.update(coherences)

    # TODO: add actual checks and return boolean
    # indicating whether or not data has passed them
    return True
