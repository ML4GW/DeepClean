import time
from pathlib import Path
from typing import Iterable, Optional, Union

import tritonclient.grpc as triton
from cleaner.dataloader import get_data_generators
from cleaner.writer import ASDRMonitor, Writer
from microservice.deployment import Deployment
from tritonclient.utils import InferenceServerException

from deepclean.infer.callback import Callback, State
from deepclean.logging import logger
from deepclean.utils.channels import ChannelList, get_channels
from hermes.aeriel.client import InferenceClient
from typeo import scriptify


def wait(url):
    client = triton.InferenceServerClient(url)

    # first wait for server to come online
    start_time = time.time()
    while True:
        try:
            live = client.is_server_live()
        except InferenceServerException:
            time.sleep(1)
        else:
            if live:
                break

        elapsed = (time.time() - start_time) // 1
        if not elapsed % 10:
            logger.info(
                f"Waiting for server to come online, {elapsed}s elapsed"
            )

    start_time = time.time()
    while not client.is_model_ready("deepclean-stream"):
        time.sleep(1)

        elapsed = (time.time() - start_time) // 1
        if not elapsed % 10:
            logger.info(
                "Waiting for streaming model to "
                f"come online, {elapsed}s elapsed"
            )

    start_time = time.time()
    while True:
        metadata = client.get_model_metadata("deepclean")
        versions = list(map(int, metadata.versions))
        if max(versions) > 1:
            break

        time.sleep(1)
        elapsed = (time.time() - start_time) // 1
        if not elapsed % 10:
            logger.info(
                "Waiting for first DeepClean model to "
                f"come online, {elapsed}s elapsed"
            )


@scriptify
def main(
    # IO args
    run_directory: Path,
    data_directory: Path,
    data_field: str,
    export_endpoint: str,
    # Data args
    sample_rate: float,
    frame_length: float,
    inference_sampling_rate: float,
    batch_size: int,
    inference_rate: float,
    channels: ChannelList,
    freq_low: Union[float, Iterable[float]],
    freq_high: Union[float, Iterable[float]],
    # Triton args
    triton_endpoint: str,
    model_name: str,
    sequence_id: int = 1001,
    # Misc args
    max_latency: Optional[float] = None,
    start_first: bool = False,
    timeout: Optional[float] = None,
    memory: float = 10,
    look_ahead: float = 0.5,
    verbose: bool = False,
) -> None:
    """
    Iterate through directories of witness and strain
    data and produced DeepClean-ed frames of strain data
    from them. Directories will be traversed in chronological
    order, using the filenames to infer the timestamps of
    each frame file.

    For true replay streams, directories
    will be monitored for new files, eventually raising an error
    after `timeout` seconds if no new files are created in the
    directory and `timeout` is not `None`.

    Args:
        witness_data_dir:
            Directory containing frames of witness channels
        strain_data_dir:
            Directory containing frames of corresponding
            strain channel data
        write_dir: Directory to which to write cleaned frame files
        sample_rate:
            The rate at which to sample witness and strain data, in Hz
        kernel_length:
            The length of the input kernel passed to DeepClean in seconds
        inference_sampling_rate:
            The rate at which to sample kernels from the witness
            and strain timeseries, in Hz. Must be less than or
            equal to `sample_rate`.
        inference_rate:
            The maximum rate at which to send inference requests
            to the server in Hz
        channels:
            Either a list of channels to leverage, with the strain
            channel first, or a path to a file containing such a
            list separated by newlines.
        freq_low:
            The low-end of each frequency band over which to
            bandpass filter the noise estimates before subtracting
            them from the strain channel, in Hz. If only a single
            float is specified, this will be used as the low-end of a
            single frequency band. If multiple are specified, they
            must be sorted in ascending order and not overlap with
            any other bands.
        freq_high:
            The high-end of each frequency band over which to
            bandpass filter the noise estimates before subtracting
            them from the strain channel, in Hz. If only a single
            float is specified, this will be used as the high-end of
            a single frequency band. The number of bands must match
            those specified in `freq_low`, and the same restrictions
            around ordering apply as well.
        url:
            Address at which server is accepting gRPC inference requests
        model_name:
            The name of the model to request for inference
        model_version:
            The version of the model to request for inference
        sequence_id:
            The ID to assign to this sequence of inference requests
            to maintain a streaming state on the server
        max_latency:
            The maximum amount of latency that server-side aggregation
            can introduce into the processing stream. I.e., the number
            of aggregation steps will be
            `floor(max_latency * inference_sampling_rate)`. If left
            as `None`, it will be assumed that the server is performing
            no aggregation.
        start_first:
            Whether to start with the first timestamped frame in the
            witness and strain data directories, or the last. The latter
            is useful for true streaming replays where getting a good
            estimate of latency is important.
        timeout:
            Once the witness directory has been traversed, this specifies
            the amount of time to wait for a new file to be generated
            before raising an error. If left as `None`, the process will
            wait indefinitely.
        memory:
            The amount of past data to keep in memory for postprocessing
            the current frame. Useful for avoiding filtering edge effects.
        look_ahead:
            The amount of future data required for postprocessing the
            current frame. Useful for avoiding filtering edge effects.
        verbose:
            Flag indicating whether to log at DEBUG or INFO levels
    """

    deployment = Deployment(run_directory)
    log_file = deployment.log_directory / "infer.log"
    logger.set_logger("DeepClean infer", log_file, verbose)
    channels = get_channels(channels)

    wait(triton_endpoint)
    witness_it, strain_it = get_data_generators(
        data_directory,
        data_field,
        channels,
        sample_rate=sample_rate,
        inference_sampling_rate=inference_sampling_rate,
        batch_size=batch_size,
        start_first=start_first,
        timeout=timeout,
    )

    # now create a file writer which takes the
    # responses returned by the server and turns them
    # into a timeseries of noise predictions which can
    # be subtracted from strain data. Use the initial
    # timestamp utilized by the crawler to crawl through
    # the strain data directory in the same order
    if max_latency is None:
        aggregation_steps = 0
    else:
        aggregation_steps = int(max_latency * inference_sampling_rate)

    states = {}
    for model in ["production", "canary"]:
        # TODO: infer this from server
        name = f"aggregator-{model}-output_stream"
        state = State(
            model,
            frame_length,
            memory=memory,
            filter_pad=look_ahead,
            sample_rate=sample_rate,
            inference_sampling_rate=inference_sampling_rate,
            batch_size=batch_size,
            aggregation_steps=aggregation_steps,
            freq_low=freq_low,
            freq_high=freq_high,
        )
        states[name] = state

    write_dir = deployment.frame_directory
    monitor = ASDRMonitor(
        buffer_length=8, freq_low=freq_low, freq_high=freq_high, fftlength=2
    )
    writer = Writer(write_dir, strain_it, monitor, export_endpoint)
    callback = Callback(writer, **states)

    # finally create an inference client which will stream
    # the updates produced by the frame iterator to the
    # inference server and pass the responses to the writer
    # as a callback in a separate thread
    client = InferenceClient(
        address=triton_endpoint,
        model_name=model_name,
        model_version=-1,
        callback=callback,
    )

    # start a streaming connection to the server that passes
    # server responses to the writer in a callback thread
    with client:
        for request_id, (x, sequence_end) in enumerate(witness_it):
            logger.debug(f"Sending request {request_id}")
            client.infer(
                x,
                request_id,
                sequence_id,
                sequence_start=request_id == 0,
                sequence_end=sequence_end,
            )
            if request_id == 0:
                callback.block(0, client.get)
            time.sleep(0.95 / inference_rate)

            # check if any files have been written or if the
            # client's callback thread has raised any exceptions
            response = client.get()
            if response is not None:
                fname, latency = response
                logger.info(f"Wrote file {fname} with latency {latency}s")


if __name__ == "__main__":
    main()
