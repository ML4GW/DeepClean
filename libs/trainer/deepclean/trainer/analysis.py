from pathlib import Path
from typing import Iterable, Optional, Sequence

import h5py
import numpy as np
import torch

from deepclean.trainer.utils import DataGenerator


def write_analysis(group, name, **data):
    subgroup = group.create_group(name)
    for key, value in data.items():
        subgroup.create_dataset(key, data=value)


def analyze_model(
    output_directory: Path,
    nn: torch.nn.Module,
    welch: torch.nn.Module,
    train_dataset: DataGenerator,
    valid_dataset: Optional[DataGenerator] = None,
    percentiles: Sequence[float] = [5, 25, 50, 75, 95],
) -> None:
    with h5py.File(output_directory / "history.h5", "a") as f:
        grads, coheres = analyze_dataset(train_dataset, nn, welch, percentiles)
        group = f.create_group("analysis")
        group.attrs["percentiles"] = percentiles
        write_analysis(group, "train", gradient=grads, coherence=coheres)

        if valid_dataset is not None:
            grads, coheres = analyze_dataset(
                valid_dataset, nn, welch, percentiles
            )
            write_analysis(group, "valid", gradient=grads, coherence=coheres)


def analyze_dataset(
    dataset: DataGenerator,
    nn: torch.nn.Module,
    welch: torch.nn.Module,
    percentiles: Iterable[float] = [5, 25, 50, 75, 95],
):
    coherences, gradients = [], []
    for i, (x, y) in enumerate(dataset):
        if i == 0:
            welch.window = welch.window.to(x.device)

        x = torch.autograd.Variable(x, requires_grad=True)
        noise_prediction = nn(x)

        # measure magnitude of gradient of output with
        # respect to the input, averaged over kernel
        cost = noise_prediction.mean()
        cost.backward(retain_graph=True)
        grads = x.grad.abs().mean(axis=-1)

        # now that we don't need gradients anymore,
        # detach everything from the computation graph
        # to save both time and memory
        grads = grads.detach().cpu().numpy()
        x, y = x.detach(), y.detach()

        pxx = welch(x)
        pyy = welch(y)[:, None]
        pxy = welch(x, y)
        coherence = pxy.abs() ** 2 / pxx / pyy

        gradients.append(grads)
        coherences.append(coherence.cpu().numpy())

    gradients = np.concatenate(gradients, axis=0)
    coherences = np.concatenate(coherences, axis=0)

    gradients = np.percentile(gradients, percentiles, axis=0)
    coherences = np.percentile(coherences, percentiles, axis=0)
    return gradients, coherences
