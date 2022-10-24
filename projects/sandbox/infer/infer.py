# TODO: complete documentation
import time
from pathlib import Path
from typing import List, Optional

import numpy as np

from deepclean.gwftools.channels import ChannelList, get_channels
from deepclean.gwftools.io import find
from deepclean.infer.write import FrameWriter
from deepclean.logging import configure_logging
from deepclean.signal.filter import FREQUENCY, BandpassFilter
from hermes.aeriel.client import InferenceClient
from hermes.aeriel.serve import serve
from hermes.typeo import typeo


def strain_iterator(
    strain: np.ndarray, frame_length: int, sample_rate: float, t0: int
) -> np.ndarray:
    frame_size = int(frame_length * sample_rate)
    num_frames = len(strain) // frame_size
    i = 0
    while i < num_frames:
        tstamp = int(t0 + i * frame_length)
        fname = f"STRAIN-{tstamp}-{frame_length}.gwf"
        yield strain[i * frame_size : (i + 1) * frame_size], Path(fname)
        i += 1


@typeo
def main(
    url: str,
    model_repo_dir: str,
    model_name: str,
    channels: ChannelList,
    t0: int,
    duration: int,
    data_path: Path,
    output_directory: Path,
    kernel_length: float,
    sample_rate: float,
    batch_size: int,
    inference_sampling_rate: float,
    inference_rate: float,
    max_latency: float,
    memory: float,
    look_ahead: float,
    freq_low: FREQUENCY,
    freq_high: FREQUENCY,
    model_version: int = -1,
    sequence_id: int = 1001,
    force_download: bool = False,
    verbose: bool = False,
    gpus: Optional[List[int]] = None,
):
    """
    Serve up the models from the indicated model repository
    for inference using Triton and stream witness data taken
    from one second-long frame files to clean the corresponding
    strain data in an online fashion.

    Args:
        url:
            Address at which Triton service is being hosted and
            to which to send requests, including port
        model_repo_dir:
            Directory containing models to serve for inference
        model_name:
            Model to which to send streaming inference requests
        output_directory:
            Directory to save logs and cleaned frames
        channels:
            A list of channel names used by DeepClean, with the
            strain channel first, or the path to a text file
            containing this list separated by newlines
        kernel_length:
            The length, in seconds, of the input to DeepClean
        stride_length:
            The length, in seconds, between kernels sampled
            at inference time. This, along with the `sample_rate`,
            dictates the size of the update expected at the
            snapshotter model
        sample_rate:
            Rate at which the input kernel has been sampled, in Hz
        max_latency:
            The maximum amount of time, in seconds, allowed during
            inference to wait for overlapping predictcions for
            online averaging. For example, if the `stride_length`
            is 0.002s and `max_latency` is 0.5s, then output segments
            will be averaged over 250 overlapping kernels before
            being streamed back from the server. This means there is
            a delay of `max_latency` (or the greatest multiple
            of `stride_length` that is less than `max_latency`) seconds
            between the start timestamp of the update streamed to
            the snapshotter and the resulting prediction returned by
            the ensemble model. The online averaging model being served
            by Triton should have been instantiated with this same value.
        memory:
            The number of seconds of past data to use when filtering
            a frame's worth of noise predictions before subtraction to
            avoid edge effects
        look_ahead:
            The number of seconds of _future_ data required to be available
            before filtering a frame's worth of noise predictions before
            subtraction to avoid edge effects
        freq_low:
            Lower limit(s) of frequency range(s) over which to filter
            noise estimates, in Hz. Specify multiple to filter over
            multiple ranges. In this case, must be same length
            as `freq_high`.
        freq_high:
            Upper limit(s) of frequency range(s) over which to filter
            noise estimates, in Hz. Specify multiple to filter over
            multiple ranges. In this case, must be same length
            as `freq_low`.
        sequence_id:
            A unique identifier to give this input/output snapshot state
            on the server to ensure streams are updated appropriately
        verbose:
            If set, log at `DEBUG` verbosity, otherwise log at
            `INFO` verbosity.
        gpus:
            The indices of the GPUs to use for inference
    """

    # load the channels from a file if we specified one
    channels = get_channels(channels)
    configure_logging(output_directory / "infer.log", verbose)
    sleep = batch_size / inference_rate / inference_sampling_rate

    # launch a singularity container hosting the server and
    # take care of some data bookkeeping while we wait for
    # it to come online
    log_file = output_directory / "server.log"
    serve_ctx = serve(model_repo_dir, gpus=gpus, log_file=log_file, wait=False)
    with serve_ctx as instance:
        # load all of the test data into memory, making sure we
        # have enough to fully process the desired interval
        # (including any future data _past_ that interval)
        max_latency = max_latency or 0
        data = find(
            channels,
            t0,
            duration + max_latency + look_ahead,
            sample_rate,
            data_path=data_path,
            force_download=force_download,
        )

        # set up a lazy data iterator for the strain data
        strain = data[channels[0]]
        strain_iter = strain_iterator(strain, 1, sample_rate, t0)

        # set up a file writer to use as a callback
        # for the client to handle server responses
        callback = FrameWriter(
            strain_iter=strain_iter,
            write_dir=output_directory / "cleaned",
            channel_name=channels[0],
            inference_sampling_rate=inference_sampling_rate,
            sample_rate=sample_rate,
            batch_size=batch_size,
            postprocessor=BandpassFilter(freq_low, freq_high, sample_rate),
            memory=memory,
            look_ahead=look_ahead,
            aggregation_steps=int(max_latency * inference_sampling_rate) - 1,
        )

        # extract the non-strain channels into our input array
        X = np.stack([data[channel] for channel in channels[1:]])
        X = X.astype("float32")

        stride = int(sample_rate // inference_sampling_rate) * batch_size
        num_steps = int(X.shape[-1] // stride)

        # wait for the server to come online so we can
        # connect to it with our client, then run inference
        instance.wait()
        client = InferenceClient(
            address=url,
            model_name=model_name,
            model_version=model_version,
            callback=callback,
        )
        with client:
            for i in range(num_steps):
                x = X[:, i * stride : (i + 1) * stride]
                client.infer(
                    x,
                    request_id=i,
                    sequence_id=sequence_id,
                    sequence_start=i == 0,
                    sequence_end=i == (num_steps - 1),
                )
                time.sleep(sleep)
                callback.block(0, client.get)
            callback.block(i, client.get)


if __name__ == "__main__":
    main()
