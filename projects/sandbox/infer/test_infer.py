import time
from pathlib import Path

import h5py
import numpy as np
import toml
import torch
from tritonclient import grpc as triton

from deepclean.architectures import DeepCleanAE
from deepclean.export import PrePostDeepClean

with open("../pyproject.toml", "r") as f:
    config = toml.load(f)

channels = config["tool"]["typeo"]["base"]["channels"]
project_dir = Path("/home/alec.gunny/deepclean/results/august-o3")
fname = project_dir.parent.parent / "data" / "deepclean-1250984770-4097.h5"
chans = []
with h5py.File(fname, "r") as f:
    for channel in sorted(channels[1:]):
        chans.append(f[channel][: 4096 * 4])
x = np.stack(chans).astype("float32")
pad = np.zeros((len(chans), 4096 - 32))
padded = np.concatenate([pad, x], axis=1)

nn = DeepCleanAE(21)
nn = PrePostDeepClean(nn)
nn.eval()
nn.load_state_dict(torch.load(project_dir / "weights.pt"))

num_kernels = (padded.shape[-1] - 4096) // 32 + 1
local_results = []
for i in range(1):  # num_kernels):
    kernel = torch.Tensor(padded[None, :, i * 32 : i * 32 + 4096])
    with torch.no_grad():
        y = nn(kernel).cpu().numpy()[0]
    local_results.append(y)

# averaged_local_results = np.zeros((4096 * 3,))
# start, i = 0, 0
# while start < (4096 * 3 - 32):
#     j = min(i + 1, 64)
#     for k in range(j):
#         averaged_local_results[start:start + 32] = (
#             local_results[i + k][k * 32:(k + 1) * 32] / j
#         )
#     start += 32
#     i += 1

server_results = []


def callback(result, error):
    server_results.append(result.as_numpy("output_stream")[0])


client = triton.InferenceServerClient("localhost:8001")
input = triton.InferInput(
    name="snapshot_update", shape=(1, 21, 2048), datatype="FP32"
)
with client:
    client.start_stream(callback=callback)
    for i in range(3):
        update = x[None, :, i * 2048 : (i + 1) * 2048]
        input.set_data_from_numpy(update)
        client.async_stream_infer(
            "deepclean-stream",
            model_version="1",
            inputs=[input],
            sequence_id=1001,
            sequence_start=i == 0,
            sequence_end=i == 7,
        )

while len(server_results) < 3:
    time.sleep(1e-3)

# server_results = np.concatenate(server_results)[2048:]
