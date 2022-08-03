from typing import Iterable, Tuple

import numpy as np
import torch

from deepclean.trainer.criterion import TorchWelch


def analyze_model(
    dataset: Iterable[Tuple[torch.Tensor, torch.Tensor]],
    nn: torch.nn.Module,
    sample_rate: float,
    fftlength: float = 2,
):
    welch = TorchWelch(sample_rate, fftlength, average="median", device="cuda")
    asds, gradients, errors = [], [], []
    for x, y in dataset:
        x = torch.autograd.Variable(x, requires_grad=True)
        noise_prediction = nn(x)

        # measure magnitude of gradient of output with
        # respect to the input, averaged over kernel
        cost = noise_prediction.mean()
        cost.backward(retain_graph=True)
        grads = x.grad[0].abs().mean(axis=-1)

        grads = grads.detach().cpu().numpy()
        noise_prediction = noise_prediction.detach()
        x, y = x.detach(), y.detach()

        # measure error as a function of position in kernel
        residual = y - noise_prediction
        errs = torch.abs(residual) / torch.abs(noise_prediction)
        errs = errs.cpu().numpy()

        # measure asd of all channels
        X = torch.cat([y[:, None], x], axis=1)
        X = X.reshape(len(x) * (x.shape[1] + 1), -1)
        psd = welch(X)
        psd = psd.reshape(len(x), x.shape[1] + 1, -1)
        psd = psd.cpu().numpy()

        asds.append(psd**0.5)
        gradients.append(grads)
        errors.append(errs)

    gradients = np.concatenate(gradients, axis=0)
    errors = np.concatenate(errors, axis=0)
    asds = np.concatenate(asds, axis=0)
    return gradients, errors, asds
