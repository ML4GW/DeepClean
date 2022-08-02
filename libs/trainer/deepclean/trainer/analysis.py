from typing import Iterable, Tuple

import numpy as np
import scipy
import torch


def analyze_model(
    dataset: Iterable[Tuple[torch.Tensor, torch.Tensor]],
    nn: torch.Module,
    sample_rate: float,
    fftlength: float = 2,
):
    asds, gradients, errors = [], [], []
    for x, y in dataset:
        x = torch.autograd.Variable(x, requires_grad=True)
        noise_prediction = nn(x)

        # measure error as a function of position in kernel
        residual = y - noise_prediction
        errs = torch.abs(residual) / torch.abs(noise_prediction)

        # measure magnitude of gradient of output with
        # respect to the input, averaged over kernel
        cost = noise_prediction.mean()
        cost.backward(retain_graph=True)
        grads = x.grad[0].abs().mean(axis=-1)

        # measure asd of all channels
        X = torch.cat([y[:, None], x], axis=1)
        X = X.detach().cpu().numpy()
        asd = (
            scipy.signal.welch(
                X,
                fs=sample_rate,
                nperseg=int(fftlength * sample_rate),
                window="hann",
                average="median",
                axis=-1,
            )
            ** 0.5
        )

        asds.append(asd)
        gradients.append(grads.detach().cpu().numpy())
        errors.append(errs.detach().cpu().numpy())

    outputs = [asds, gradients, errors]
    outputs = [np.concatenate(i, axis=0) for i in outputs]
    return outputs
