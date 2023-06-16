from typing import List, Optional, Union

import numpy as np
import torch

from deepclean.utils.filtering import Frequency, normalize_frequencies
from ml4gw.transforms import SpectralDensity


class PSDLoss(torch.nn.Module):
    """Compute the power spectrum density (PSD) loss, defined
    as the average over frequency of the PSD ratio"""

    def __init__(
        self,
        sample_rate: float,
        fftlength: float,
        asd: bool = False,
        overlap: Optional[float] = None,
        freq_low: Union[Frequency, None] = None,
        freq_high: Union[Frequency, None] = None,
    ):
        super().__init__()

        fast = False
        if freq_low is not None and freq_high is not None:
            freq_low, freq_high = normalize_frequencies(freq_low, freq_high)

            # since we specified frequency ranges, build a mask
            # to zero out the frequencies we don't care about
            dfreq = 1 / fftlength
            freqs = np.arange(0.0, sample_rate / 2 + dfreq, dfreq)
            mask = np.zeros_like(freqs, dtype=bool)
            for low, high in zip(freq_low, freq_high):
                mask |= (low <= freqs) & (freqs < high)

            # if we don't care about the values in the two lowest
            # frequency bins, then we're safe to use a faster
            # implementation of the welch transform
            if not mask[:2].any():
                fast = True
            mask = torch.tensor(mask, dtype=torch.bool)
            self.register_buffer("mask", mask)
        elif freq_low is None and freq_high is None:
            # no frequencies were specified, so ignore the mask
            self.mask = None
        else:
            # one was specified and the other wasn't, so build
            # a generic error that will populate with the
            # appropriate values at error-time
            raise ValueError(
                "If '{}' is specified, '{}' must be specified as well".format(
                    "freq_high" if freq_low is None else "freq_low",
                    "freq_low" if freq_low is None else "freq_high",
                )
            )

        self.welch = SpectralDensity(
            sample_rate,
            fftlength,
            overlap,
            average="mean",
            fast=fast,
        )
        self.asd = asd

    def forward(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # Calculate the PSD of the residual and the target
        psd_res = self.welch(y - y_hat)
        psd_target = self.welch(y)

        if self.mask is not None:
            # mask out these arrays rather than the ratio since
            # fp32 precision in the asd calculation can 0-out
            # some psd values in the target array, especially
            # if they've been filtered beforehand. This leads
            # to nan values that propogate back through the gradient
            psd_res = torch.masked_select(psd_res, self.mask)
            psd_res = psd_res.reshape(len(y), -1)

            psd_target = torch.masked_select(psd_target, self.mask)
            psd_target = psd_target.reshape(len(y), -1)

        ratio = psd_res / psd_target
        if self.asd:
            ratio = ratio**0.5
        return ratio.mean()


class CompositePSDLoss(torch.nn.Module):
    """PSD + MSE Loss with weight"""

    def __init__(
        self,
        alpha: float,
        sample_rate: Optional[float] = None,
        fftlength: Optional[float] = None,
        overlap: Optional[float] = None,
        asd: bool = False,
        freq_low: Union[float, List[float], None] = None,
        freq_high: Union[float, List[float], None] = None,
    ):
        super().__init__()
        if not 0 <= alpha <= 1:
            raise ValueError("Alpha value '{}' out of range".format(alpha))
        self.alpha = alpha

        if alpha > 0:
            if sample_rate is None or fftlength is None:
                raise ValueError(
                    "Must specify both 'sample_rate' and "
                    "'fftlength' if alpha > 0"
                )

            self.psd_loss = PSDLoss(
                sample_rate=sample_rate,
                freq_low=freq_low,
                freq_high=freq_high,
                fftlength=fftlength,
                overlap=overlap,
                asd=asd,
            )
        self.mse_loss = torch.nn.MSELoss()

    def forward(self, pred, target):
        if self.alpha == 0:
            return self.mse_loss(pred, target)
        if self.alpha == 1:
            return self.psd_loss(pred, target)

        psd_loss = self.psd_loss(pred, target)
        mse_loss = self.mse_loss(pred, target)

        return self.alpha * psd_loss + (1 - self.alpha) * mse_loss
