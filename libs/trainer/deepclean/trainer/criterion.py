from typing import List, Optional, Union

import numpy as np
import torch
import torch.nn as nn
from scipy.signal.spectral import _median_bias

from deepclean.signal.filter import normalize_frequencies


def _validate_shapes(x: torch.Tensor, y: torch.Tensor):
    # acceptable combinations of shapes:
    # x: time, y: time
    # x: channel x time, y: time OR channel x time
    # x: batch x channel x time, y: batch x channel x time OR batch x time
    if x.shape[-1] != y.shape[-1]:
        raise ValueError(
            "Time dimensions of x and y tensors must "
            "be the same, found {} and {}".format(x.shape[-1], y.shape[-1])
        )
    elif x.ndim == 1 and not y.ndim == 1:
        raise ValueError(
            "Can't compute cross spectral density of "
            "1D tensor x with {}D tensor y".format(y.ndim)
        )
    elif x.ndim > 1 and y.ndim == x.ndim:
        if not y.shape == x.shape:
            raise ValueError(
                "If x and y tensors have the same number "
                "of dimensions, shapes must fully match. "
                "Found shapes {} and {}".format(x.shape, y.shape)
            )
    elif x.ndim > 1 and y.ndim != (x.ndim - 1):
        raise ValueError(
            "Can't compute cross spectral density of "
            "tensors with shapes {} and {}".format(x.shape, y.shape)
        )
    elif x.ndim > 2 and y.shape[0] != x.shape[0]:
        raise ValueError(
            "If x is a 3D tensor and y is a 2D tensor, "
            "0th batch dimensions must match, but found "
            "values {} and {}".format(x.shape[0], y.shape[0])
        )


def _median(x, axis):
    return torch.quantile(x, q=0.5, axis=axis)


def _fast_csd(
    x: torch.Tensor,
    nperseg: int,
    nstride: int,
    window: torch.Tensor,
    scale: torch.Tensor,
    average: str,
    y: Optional[torch.Tensor] = None,
):
    # for fast implementation, use the built-in stft
    # module along the last dimension and let torch
    # handle the heavy lifting under the hood
    if x.ndim > 2:
        # stft only works on 2D input, so roll the
        # channel dimension out along the batch
        batch_size = x.shape[0]
        num_channels = x.shape[1]
        x = x.reshape(-1, x.shape[-1])

        if y is not None and y.ndim == 3:
            y = y.reshape(-1, x.shape[-1])
    else:
        batch_size = None

    x = x - x.mean(axis=-1, keepdims=True)
    fft = torch.stft(
        x,
        n_fft=nperseg,
        hop_length=nstride,
        window=window,
        normalized=False,
        center=False,
        return_complex=True,
    )
    if y is not None:
        y = y - y.mean(axis=-1, keepdims=True)
        y_fft = torch.stft(
            y,
            n_fft=nperseg,
            hop_length=nstride,
            window=window,
            normalized=False,
            center=False,
            return_complex=True,
        )
        if batch_size is not None and fft.shape[0] > y_fft.shape[0]:
            nfreq = nperseg // 2 + 1
            fft = fft.reshape(batch_size, num_channels, nfreq, -1)
            y_fft = y_fft[:, None]

        fft = torch.conj(fft) * y_fft
    else:
        fft = fft.abs() ** 2

    stop = None if nperseg % 2 else -1
    if x.ndim == 1:
        fft[1:stop] *= 2
    elif fft.ndim < 4:
        fft[:, 1:stop] *= 2
    else:
        fft[:, :, 1:stop] *= 2
    fft *= scale

    if average == "mean":
        fft = fft.mean(axis=-1)
    else:
        bias = _median_bias(fft.shape[-1])
        if y is not None:
            real_median = _median(fft.real, -1)
            imag_median = 1j * _median(fft.imag, -1)
            fft = real_median + imag_median
        else:
            fft = _median(fft, -1)
        fft /= bias

    if fft.ndim == 2 and batch_size is not None:
        fft = fft.reshape(batch_size, num_channels, -1)
    return fft


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
        self.nstride = self.nperseg - int(overlap * sample_rate)

        # do we allow for arbitrary windows?
        self.window = torch.hann_window(self.nperseg).to(device)

        # scale corresponds to "density" normalization, worth
        # considering adding this as a kwarg and changing this calc
        self.scale = 1.0 / (sample_rate * (self.window**2).sum())

        if average not in ("mean", "median"):
            raise ValueError(
                f'average must be "mean" or "median", got {average} instead'
            )
        self.average = average
        self.device = device
        self.fast = fast

    def forward(self, x: torch.Tensor, y: Optional[torch.Tensor] = None):
        if x.shape[-1] < self.nperseg:
            raise ValueError(
                "Number of samples {} in input x is insufficient "
                "for number of fft samples {}".format(
                    x.shape[-1], self.nperseg
                )
            )
        elif x.ndim > 3:
            raise ValueError(
                f"Can't perform welch transform on tensor with shape {x.shape}"
            )

        if y is not None:
            _validate_shapes(x, y)

        if self.fast:
            return _fast_csd(
                x,
                y=y,
                nperseg=self.nperseg,
                nstride=self.nstride,
                window=self.window,
                scale=self.scale,
                average=self.average,
            )

        if y is not None:
            raise NotImplementedError

        # for non-fast implementation, we need to unfold
        # the tensor along the time dimension ourselves
        # to detrend each segment individually, so start
        # by converting x to a 4D tensor so we can use
        # torch's Unfold op
        if x.ndim == 1:
            reshape = []
            x = x[None, None, None, :]
        elif x.ndim == 2:
            reshape = [len(x)]
            x = x[None, :, None, :]
        elif x.ndim == 3:
            reshape = list(x.shape[:-1])
            x = x[:, :, None, :]

        # calculate the number of segments and trim x along
        # the time dimensions so that we can unfold it exactly
        num_segments = (x.shape[-1] - self.nperseg) // self.nstride + 1
        stop = (num_segments - 1) * self.nstride + self.nperseg
        x = x[..., :stop]

        # unfold x into overlapping segments and detrend and window
        # each one individually before computing the rfft. Unfold
        # will produce a batch x (num_channels * num_segments) x nperseg
        # shaped tensor
        unfold_op = torch.nn.Unfold(
            (1, num_segments), dilation=(1, self.nstride)
        )
        x = unfold_op(x)
        x = x - x.mean(axis=-1, keepdims=True)
        x *= self.window

        # after the fft, we'll have a
        # batch x (num_channels * num_segments) x nfreq
        # sized tensor
        fft = torch.fft.rfft(x, axis=-1).abs() ** 2

        if self.nperseg % 2:
            fft[..., 1:] *= 2
        else:
            fft[..., 1:-1] *= 2
        fft *= self.scale

        # unfold the batch and channel dimensions back
        # out if there were any to begin with, putting
        # the segment dimension as the second to last
        reshape += [num_segments, -1]
        fft = fft.reshape(*reshape)

        if self.average == "mean":
            return fft.mean(axis=-2)
        else:
            bias = _median_bias(num_segments)
            return _median(fft, -2) / bias


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
            self.mask = torch.tensor(mask, dtype=torch.bool).to(device)
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

        self.welch = TorchWelch(
            sample_rate,
            fftlength,
            overlap,
            average="median",
            device=device,
            fast=fast,
        )
        self.asd = asd

    def forward(self, pred, target):
        # Calculate the PSD of the residual and the target
        psd_res = self.welch(target - pred)
        psd_target = self.welch(target)

        if self.mask is not None:
            # mask out these arrays rather than the ratio since
            # fp32 precision in the asd calculation can 0-out
            # some psd values in the target array, especially
            # if they've been filtered beforehand. This leads
            # to nan values that propogate back through the gradient
            psd_res = torch.masked_select(psd_res, self.mask)
            psd_res = psd_res.reshape(len(pred), -1)

            psd_target = torch.masked_select(psd_target, self.mask)
            psd_target = psd_target.reshape(len(pred), -1)

        ratio = psd_res / psd_target
        if self.asd:
            ratio = ratio ** (0.5)
        return ratio.mean()


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
