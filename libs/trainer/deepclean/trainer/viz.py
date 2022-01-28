import os
from typing import TYPE_CHECKING, Callable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

if TYPE_CHECKING:
    from deepclean.trainer.dataloader import ChunkedTimeSeriesDataset


def plot_data_asds(
    dataloader: "ChunkedTimeSeriesDataset",
    sample_rate: float,
    write_dir: str,
    welch: Callable,
    channels: List[str],
    x_range: Optional[Tuple[float]] = None,
    model: Optional[torch.nn.Module] = None,
    postprocessor: Optional[Callable] = None,
):
    os.makedirs(write_dir, exist_ok=True)

    X_asd, y_asd, res_asd, N = 0, 0, 0, 0
    for X, y in dataloader:
        # compute predictions using X
        # if we have a model to do it with
        if model is not None:
            model.eval()
            with torch.no_grad():
                pred = model(X)

            # possibly postprocess the predictions
            if postprocessor is not None:
                pred = pred.cpu().numpy()
                pred = postprocessor(pred, inverse=True)
                pred = torch.tensor(pred, device="cuda")

            # now compute the PSD of the residual
            residual = y - pred
            residual_welch = welch(residual).mean(axis=0)
        else:
            residual_welch = None

        # compute PSDs of each of the X witness channels in X
        batch_size = X.shape[0]
        X = X.view(-1, X.shape[-1])
        X = welch(X)
        X = X.view(batch_size, -1, X.shape[-1])
        X = X.mean(axis=0)

        # compute the PSD of the strain channel
        y = welch(y).mean(axis=0)

        # now update the averages in an online fashion
        factor = len(X) / (N + len(X))
        X_asd -= (X_asd - X) * factor
        y_asd -= (y_asd - y) * factor

        if model is not None:
            res_asd -= (res_asd - residual_welch) * factor
        N += len(X)

    # move these asds back to the CPU/numpy
    X_asd = np.sqrt(X_asd.cpu().numpy())
    y_asd = np.sqrt(y_asd.cpu().numpy())
    if model is not None:
        res_asd = np.sqrt(res_asd.cpu().numpy())

    asds = np.concatenate([y_asd[None], X_asd], axis=0)
    freqs = np.linspace(0, sample_rate / 2, X_asd.shape[-1])

    if x_range is not None:
        x_min, x_max = x_range
        mask = (x_min < freqs) & (freqs < x_max)
    else:
        mask = np.ones_like(freqs, dtype=np.bool)

    for channel, asd in zip(channels, asds):
        fig, ax = plt.subplots()
        ax.set_xlabel("Frequency [Hz]")
        ax.set_ylabel("ASD [Hz$^{-\\frac{1}{2}}$]")
        ax.set_yscale("log")
        ax.set_title(str(channel))

        ax.plot(freqs[mask], asd[mask])
        if model is not None and channel == channels[0]:
            ax.plot(freqs[mask], res_asd[mask], label="Cleaned")
            ax.legend()
        fig.savefig(os.path.join(write_dir, f"{channel}.png"))
        plt.close(fig)
