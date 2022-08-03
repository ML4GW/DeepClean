import logging
from typing import List, Optional, Union

import numpy as np
import torch
import torch.nn as nn

from deepclean.signal.filter import normalize_frequencies


class TorchWelch(nn.Module):
    def __init__(
        self,
        sample_rate: float,
        fftlength: float,
        overlap: Optional[float] = None,
        average: str = "mean",
        device: str = "cpu",
        fast: bool = False,
    ):
        if overlap is None:
            overlap = fftlength / 2
        elif overlap >= fftlength:
            raise ValueError(
                "Can't have overlap {} longer than fftlength {}".format(
                    overlap, fftlength
                )
            )

        super().__init__()

        self.nperseg = int(fftlength * sample_rate)
        self.noverlap = int(overlap * sample_rate)
        self.window = torch.hann_window(self.nperseg).to(device)
        self.scale = 1.0 / (sample_rate * (self.window**2).sum())

        if average not in ("mean", "median"):
            raise ValueError(
                f'average must be "mean" or "median", got {average} instead'
            )
        self.average = average
        self.device = device
        self.fast = fast

    def forward(self, x):
        if x.shape[-1] < self.nperseg:
            raise ValueError(
                "Not enough samples {} in input for number "
                "of fft samples {}".format(x.shape[-1], self.nperseg)
            )
        elif x.ndim > 3:
            raise ValueError(
                f"Can't perform welch transform on tensor with shape {x.shape}"
            )

        if self.fast:
            if x.ndim > 2:
                batch_size = x.shape[0]
                num_channels = x.shape[1]
                x = x.reshape(-1, x.shape[-1])
            else:
                batch_size = None

            x = x - x.mean(axis=-1, keepdims=True)
            fft = (
                torch.stft(
                    x,
                    n_fft=8192,
                    hop_length=4096,
                    window=self.window,
                    normalized=False,
                    center=False,
                    return_complex=True,
                ).abs()
                ** 2
            )

            if self.nperseg % 2:
                fft[:, 1:] *= 2
            else:
                fft[:, 1:-1] *= 2
            fft *= self.scale

            if self.average == "mean":
                fft = fft.mean(axis=-1)
            else:
                fft = fft.median(axis=-1).values

            if batch_size is not None:
                fft = fft.reshape(batch_size, num_channels, -1)
            return fft

        if x.ndim == 1:
            x = x[None, None, None, :]
            batch_size = num_channels = None
        elif x.ndim == 2:
            num_channels = len(x)
            batch_size = None
            x = x[None, :, None, :]
        elif x.ndim == 3:
            batch_size, num_channels = x.shape[:-1]
            x = x[:, :, None, :]

        nstride = self.nperseg - self.noverlap
        num_segments = (x.shape[-1] - self.nperseg) // nstride + 1
        stop = (num_segments - 1) * nstride + self.nperseg
        x = x[:, :, :, :stop]

        unfold_op = torch.nn.Unfold((1, num_segments), dilation=(1, nstride))
        x = unfold_op(x)
        x = x - x.mean(axis=-1, keepdims=True)
        x *= self.window
        fft = torch.fft.rfft(x, axis=-1).abs() ** 2

        if self.nperseg % 2:
            fft[:, :, 1:] *= 2
        else:
            fft[:, :, 1:-1] *= 2
        fft *= self.scale

        if batch_size is not None and num_channels is not None:
            fft = fft.reshape(batch_size, num_channels, num_segments, -1)
        elif num_channels is not None:
            fft = fft.reshape(num_channels, num_segments, -1)
        else:
            fft = fft.reshape(num_segments, -1)

        if self.average == "mean":
            return fft.mean(axis=-2)
        else:
            # TODO: implement median bias
            return torch.median(fft, axis=-2).values


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
        device: str = "cpu",
    ):
        super().__init__()

        fast = False
        if freq_low is not None and freq_high is not None:
            freq_low, freq_high = normalize_frequencies(freq_low, freq_high)

            # since we specified frequency ranges, build a mask
            # to zero out the frequencies we don't care about
            freqs = np.linspace(0.0, sample_rate / 2, self.welch.nfreq)
            mask = np.zeros_like(freqs, dtype=np.int64)
            for low, high in zip(freq_low, freq_high):
                in_range = (low <= freqs) & (freqs < high)
                mask[in_range] = 1

            if not mask[:2].any():
                fast = True

            self.mask = torch.Tensor(mask).to(device)
            self.N = self.mask.sum()
            logging.debug(f"Averaging over {self.N} frequency bins")
        elif freq_low is None and freq_high is None:
            # no frequencies were specified, so ignore the mask
            self.mask = self.scale = None
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

        self.welch = TorchWelch(
            sample_rate,
            fftlength,
            overlap,
            average="median",
            device=device,
            fast=fast,
        )

    def forward(self, pred, target):
        # Calculate the PSD of the residual and the target
        psd_res = self.welch(target - pred)
        psd_target = self.welch(target)

        ratio = psd_res / psd_target
        if self.asd:
            ratio = ratio ** (0.5)

        if self.mask is not None:
            ratio *= self.mask
            loss = torch.sum(ratio) / (self.N * len(pred))
        else:
            loss = torch.mean(ratio)
        return loss


class CompositePSDLoss(nn.Module):
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
        device: str = "cpu",
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
