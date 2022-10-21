from pathlib import Path

import h5py
import numpy as np
import toml
import torch

from deepclean.architectures import DeepCleanAE
from deepclean.export import PrePostDeepClean

# from hermes.quiver.streaming_input import Snapshotter
# from hermes.quiver.streaming_output import OnlineAverager

NUM_SECONDS = 11
STRIDE = 32
SAMPLE_RATE = 4096

with open("../pyproject.toml", "r") as f:
    config = toml.load(f)

channels = config["tool"]["typeo"]["base"]["channels"]
project_dir = Path("/home/alec.gunny/deepclean/results/august-o3")
fname = project_dir.parent.parent / "data" / "deepclean-1250984770-4097.h5"

chans = []
with h5py.File(fname, "r") as f:
    for channel in sorted(channels[1:]):
        chans.append(f[channel][: SAMPLE_RATE * NUM_SECONDS])

x = np.stack(chans).astype("float32")
pad = np.zeros((len(chans), SAMPLE_RATE - STRIDE))
padded = np.concatenate([pad, x], axis=1)

print("Initializing deepclean and loading in weights")
nn = DeepCleanAE(21)
nn = PrePostDeepClean(nn)
nn.eval()
nn.load_state_dict(torch.load(project_dir / "weights.pt"))

print("Running vanilla inference")
num_kernels = (padded.shape[-1] - SAMPLE_RATE) // STRIDE + 1
local_results = []
for i in range(num_kernels):
    slc = (i * STRIDE, i * STRIDE + SAMPLE_RATE)
    kernel = torch.Tensor(padded[None, :, slc])

    with torch.no_grad():
        y = nn(kernel).cpu().numpy()[0]
    local_results.append(y)

print("Averaging vanilla inference results")
averaged_local_results = np.zeros((4096 * (NUM_SECONDS - 1),))
num_steps = len(averaged_local_results) // STRIDE
num_average = SAMPLE_RATE / 2 / STRIDE
for i in range(num_steps):
    output_slice = slice(i * STRIDE, (i + 1) * STRIDE)
    for j in range(num_average):
        start = -(j + 1) * STRIDE
        stop = -j * STRIDE or None
        update = local_results[i + j][start:stop] / num_average
        averaged_local_results[output_slice] += update
