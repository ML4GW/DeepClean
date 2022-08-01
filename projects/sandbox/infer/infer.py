# TODO: complete documentation
import logging
import time
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import List, Optional

import numpy as np
from gwpy.timeseries import TimeSeries

from deepclean.gwftools.channels import ChannelList, get_channels
from deepclean.gwftools.io import find
from deepclean.infer.write import FrameWriter
from deepclean.logging import configure_logging
from deepclean.signal.filter import FREQUENCY, BandpassFilter
from hermes.aeriel.client import InferenceClient
from hermes.aeriel.serve import serve
from hermes.typeo import typeo


@contextmanager
def write_strain(
    strain: np.ndarray,
    t0: float,
    num_frames: int,
    stride: int,
    sample_rate: float,
    channel: str,
):
    # since the FrameWriter class loads strain data internally,
    # split the strain channel into 1s frames and write them to
    # a temporary directory for the writer to load them
    strain = strain[: int(num_frames * sample_rate)]
    strain = np.split(strain, num_frames)
    with TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        for i, y in enumerate(strain):
            ts = TimeSeries(y, dt=1 / sample_rate, channel=channel)
            tstamp = int(t0 + i)
            ts.write(tmpdir / f"STRAIN-{tstamp}-1.gwf")
        yield tmpdir


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
    inference_sampling_rate: float,
    sample_rate: float,
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

    # launch a singularity container hosting the server and
    # take care of some data bookkeeping while we wait for
    # it to come online
    with serve(
        model_repo_dir,
        gpus=gpus,
        log_file=output_directory / "server.log",
        wait=False,
    ) as instance:
        # load all of the test data into memory
        data = find(
            channels,
            t0,
            duration + max_latency + look_ahead,
            sample_rate,
            data_path=data_path,
            force_download=force_download,
        )

        # extract the non-strain channels into our input array
        X = np.stack([data[channel] for channel in channels[1:]])
        stride = int(sample_rate // inference_sampling_rate)
        num_steps = int(X.shape[-1] // stride)

        # since the FrameWriter class loads strain data internally,
        # split the strain channel into 1s frames and write them to
        # a temporary directory for the writer to load them
        y = data[channels[0]]
        with write_strain(
            y, t0, duration, stride, sample_rate, channels[0]
        ) as tmpdir:
            # set up a file writer to use as a callback
            # for the client to handle server responses
            if max_latency is None:
                aggregation_steps = 0
            else:
                aggregation_steps = int(max_latency * inference_sampling_rate)
            writer = FrameWriter(
                tmpdir,
                output_directory / "cleaned",
                channel_name=channels[0],
                inference_sampling_rate=inference_sampling_rate,
                sample_rate=sample_rate,
                t0=int(t0),
                postprocessor=BandpassFilter(freq_low, freq_high, sample_rate),
                memory=memory,
                look_ahead=look_ahead,
                aggregation_steps=aggregation_steps,
            )

            # wait for the server to come online so we can
            # connect to it with our client, then run inference
            instance.wait()
            client = InferenceClient(
                address=url,
                model_name=model_name,
                model_version=model_version,
                callback=writer,
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
                    time.sleep(0.95 / inference_sampling_rate)

                    # check if any files have been written or if the
                    # client's callback thread has raised any exceptions
                    response = client.get()
                    if response is not None:
                        fname, latency = response
                        logging.info(f"Wrote file {fname}")


if __name__ == "__main__":
    main()
