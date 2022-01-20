import logging
from typing import List, Union

import numpy as np
import torch
import torch.nn as nn


class TorchWelch(nn.Module):
    def __init__(
        self,
        sample_rate: float,
        fftlength: float,
        overlap: int,
        average: str = "mean",
        asd: bool = False,
        device: str = "cpu"
    ):
        if overlap >= fftlength:
            raise ValueError(
                "Can't have overlap {} longer than fftlength {}".format(
                    overlap, fftlength
                )
            )

        super().__init__()
        self.nperseg = int(fftlength * sample_rate)
        self.nfreq = self.nperseg // 2 + 1
        self.noverlap = int(overlap * sample_rate)
        self.window = torch.hann_window(self.nperseg).to(device)
        self.scale = 1.0 / self.window.sum()**2

        if average not in ("mean", "median"):
            raise ValueError(
                f'average must be "mean" or "median", got {average} instead'
            )
        self.average = average
        self.device = device
        self.asd = asd

    def forward(self, x):
        if len(x.shape) > 2:
            x = x.view(x.shape[0], -1)
        N, nsample = x.shape

        if nsample < self.nperseg:
            raise ValueError(
                "Not enough samples {} in input for number "
                "of fft samples".format(nsample, self.nperseg)
            )

        nstride = self.nperseg - self.noverlap
        nseg = (nsample - self.nperseg) // nstride + 1

        # Calculate the PSD
        psd = torch.zeros((nseg, N, self.nfreq)).to(self.device)

        # calculate the FFT amplitude of each segment
        # TODO: can we use torch.stft instead? Should be equivalent
        for i in range(nseg):
            seg_ts = x[:, i * nstride: i * nstride + self.nperseg]
            seg_ts = seg_ts * self.window
            seg_fd = torch.fft.rfft(seg_ts, dim=1)
            psd[i] = seg_fd.abs()**2

        if self.nperseg % 2:
            psd[:, :, 1:] *= 2
        else:
            psd[:, :, 1:-1] *= 2
        psd *= self.scale

        # taking the average
        if self.average == "mean":
            psd = torch.mean(psd, axis=0)
        elif self.average == "median":
            # TODO: account for median bias like scipy does?
            psd = torch.median(psd, axis=0)[0]

        if self.asd:
            return torch.sqrt(psd)
        return psd


class PSDLoss(nn.Module):
    """Compute the power spectrum density (PSD) loss, defined
    as the average over frequency of the PSD ratio"""

    def __init__(
        self,
        sample_rate: float,
        fftlength: float,
        overlap: float,
        asd: bool = False,
        freq_low: Union[float, List[float], None] = None,
        freq_high: Union[float, List[float], None] = None,
        device: str = "cpu"
    ):
        super().__init__()

        self.welch = TorchWelch(
            sample_rate, fftlength, overlap, asd=asd, device=device
        )
        if freq_low is not None and freq_high is not None:
            if isinstance(freq_low, (int, float)):
                freq_low = [freq_low]
                if not isinstance(freq_high, (int, float)):
                    raise ValueError(
                        "'freq_low' and 'freq_high' values {} and {} "
                        "are incompatible.".format(freq_low, freq_high)
                    )
                freq_high = [freq_high]
            else:
                try:
                    if not len(freq_low) == len(freq_high):
                        raise ValueError(
                            "Lengths of 'freq_low' and 'freq_high' {} and {} "
                            "are incompatible.".format(
                                len(freq_low), len(freq_high)
                            )
                        )
                except TypeError:
                    raise ValueError(
                        "'freq_low' and 'freq_high' values {} and {} "
                        "are incompatible.".format(freq_low, freq_high)
                    )

            freqs = np.linspace(0.0, sample_rate / 2, self.welch.nfreq)
            mask = np.zeros_like(freqs, dtype=np.int64)
            for low, high in zip(freq_low, freq_high):
                in_range = (low <= freqs) & (freqs < high)
                mask[in_range] = 1

            self.mask = torch.Tensor(mask).to(device)
            self.N = mask.sum()
            logging.debug(f"Averaging over {self.N} frequency bins")
        elif freq_low is None and freq_high is None:
            self.mask = self.N = None
        else:
            raise ValueError(
                "If '{}' is specified, '{}' must be specified as well".format(
                    "freq_high" if freq_low is None else "freq_low",
                    "freq_low" if freq_low is None else "freq_high"
                )
            )

    def forward(self, pred, target):
        # Calculate the PSD of the residual and the target
        psd_res = self.welch(target - pred)
        psd_target = self.welch(target)
        ratio = psd_res / psd_target

        if self.mask is not None:
            ratio *= self.mask
            loss = torch.sum(ratio) / (self.N * len(pred))
        else:
            loss = torch.mean(ratio)
        return loss


class CompositePSDLoss(nn.Module):
    """ PSD + MSE Loss with weight """

    def __init__(
        self,
        alpha: float,
        sample_rate: float,
        fftlength: float,
        overlap: float,
        asd: bool = False,
        freq_low: Union[float, List[float], None] = None,
        freq_high: Union[float, List[float], None] = None,
        device: str = "cpu"
    ):
        super().__init__()
        if not 0 <= alpha <= 1:
            raise ValueError(
                "Alpha value {} out of range".format(alpha)
            )
        self.alpha = alpha

        self.psd_loss = PSDLoss(
            sample_rate=sample_rate,
            freq_low=freq_low,
            freq_high=freq_high,
            fftlength=fftlength,
            overlap=overlap,
            asd=asd,
            device=device,
        )
        self.mse_loss = nn.MSELoss()

    def forward(self, pred, target):
        if self.alpha == 0:
            return self.mse_loss(pred, target)
        if self.alpha == 1:
            return self.psd_loss(pred, target)

        psd_loss = self.psd_loss(pred, target)
        mse_loss = self.mse_loss(pred, target)

        return self.alpha * psd_loss + (1 - self.alpha) * mse_loss
