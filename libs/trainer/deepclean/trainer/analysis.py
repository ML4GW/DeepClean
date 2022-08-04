from typing import Iterable, Tuple

import numpy as np
import torch

from deepclean.trainer.criterion import TorchWelch


def analyze_model(
    dataset: Iterable[Tuple[torch.Tensor, torch.Tensor]],
    nn: torch.nn.Module,
    sample_rate: float,
    fftlength: float = 2,
    percentiles: Iterable[float] = [5, 25, 50, 75, 95],
):
    welch = TorchWelch(sample_rate, fftlength, average="median", device="cuda")
    coherences, gradients = [], []
    for x, y in dataset:
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
        coherences.append(coherence.cpu().numpy())

    gradients = np.concatenate(gradients, axis=0)
    coherences = np.concatenate(coherences, axis=0)

    gradients = np.percentile(gradients, percentiles, axis=0)
    coherences = np.percentile(coherences, percentiles, axis=0)
    return gradients, coherences
