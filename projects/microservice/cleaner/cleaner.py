import logging
import time
from contextlib import contextmanager
from pathlib import Path
from queue import Empty
from typing import Iterable, Optional, Union

from hermes.stillwater import InferenceClient
from hermes.typeo import typeo

from deepclean.gwftools.channels import ChannelList, get_channels
from deepclean.infer.asynchronous import FrameLoader, FrameWriter
from deepclean.infer.frame_crawler import FrameCrawler
from deepclean.logging import configure_logging
from deepclean.signal.filter import BandpassFilter


class DummyQueue:
    def __init__(self):
        self.package = None

    def put(self, package):
        self.package = package

    def get_nowait(self):
        if self.package is None:
            raise Empty

        package = self.package
        self.package = None
        return package


@contextmanager
def stream(loader, client, writer):
    """
    Context for taking care of a lot of the business that usually
    gets dealt with in the `PipelineProcess.run` method
    """

    # hack the client callback to do the accumulation,
    # postprocessing, and writing in the callback thread
    client.out_q = writer.in_q = DummyQueue()

    def callback(result, error):
        client.callback(result, error)
        time.sleep(1e-5)

        try:
            response = writer.get_package()
            writer.process(response)
        except Exception as e:
            writer.out_q.put(str(e))

    # assign each of the processes a logger
    for p in [loader, client, writer]:
        logger = logging.getLogger(p.name)
        p.logger = logger

    # start the stream with the custom callback in a
    # try-catch to make sure everything gets cleaned up
    try:
        client.client.start_stream(callback=callback)
        yield
    finally:
        # close the stream between the client and server
        client.client.close()

        # for each process, empty all the incoming
        # and outgoing queues, then close and join them
        for p in [loader, client, writer]:
            for q in [p.in_q, p.out_q]:
                while True:
                    try:
                        q.get_nowait()
                    except (Empty, ValueError):
                        # empty means there's nothing left to get,
                        # ValueError means this queue has already
                        # been closed (likely because it's the
                        # out_q of an earlier process)
                        break

                try:
                    q.close()
                except ValueError:
                    pass
                q.join_thread()


@typeo
def main(
    # IO args
    witness_data_dir: Path,
    strain_data_dir: Path,
    write_dir: Path,
    # Data args
    sample_rate: float,
    kernel_length: float,
    inference_sampling_rate: float,
    inference_rate: float,
    channels: ChannelList,
    freq_low: Union[float, Iterable[float]],
    freq_high: Union[float, Iterable[float]],
    # Triton args
    url: str,
    model_name: str,
    model_version: int = -1,
    sequence_id: int = 1001,
    # Misc args
    max_latency: Optional[float] = None,
    start_first: bool = False,
    timeout: Optional[float] = None,
    memory: float = 10,
    look_ahead: float = 0.5,
    verbose: bool = False,
    log_file: Optional[str] = None,
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
        log_file:
            Path to a file to write logs to. If left as `None`, logs will
            only go to stdout.
    """

    configure_logging(log_file, verbose)
    channels = get_channels(channels)

    # create a frame crawler which will start at either the first
    # first or last timestep available in the witness data directory
    t0 = 0 if start_first else -1
    crawler = FrameCrawler(witness_data_dir, t0, timeout)

    # create a frame loader which will take filenames returned
    # by the crawler and load them into numpy arrays using
    # just the witness channels, then iterate through this array
    # in (1 / inference_sampling_rate)-sized updates
    loader = FrameLoader(
        inference_sampling_rate=inference_sampling_rate,
        sample_rate=sample_rate,
        channels=channels[1:],
        sequence_id=sequence_id,
        name="loader",
        join_timeout=1,
    )

    # create an inference client which take these updates
    # and use them to make inference requests to the server
    client = InferenceClient(
        url=url,
        model_name=model_name,
        model_version=model_version,
        profile=False,
        name="client",
        join_timeout=1,
    )

    # some stuff that the frame writer will need
    postprocessor = BandpassFilter(freq_low, freq_high, sample_rate)
    metadata = client.client.get_model_metadata(client.model_name)
    if max_latency is None:
        aggregation_steps = 0
    else:
        aggregation_steps = int(max_latency * inference_sampling_rate)

    # finally create a file writer which takes the
    # response returned by the server and turns them
    # into a timeseries of noise predictions which can
    # be subtracted from strain data. Use the initial
    # timestamp utilized by the crawler to crawl through
    # the strain data directory in the same order
    writer = FrameWriter(
        strain_data_dir,
        write_dir,
        channel_name=channels[0],
        inference_sampling_rate=inference_sampling_rate,
        sample_rate=sample_rate,
        t0=crawler.t0,
        postprocessor=postprocessor,
        memory=memory,
        look_ahead=look_ahead,
        aggregation_steps=aggregation_steps,
        output_name=metadata.outputs[0].name,
        name="writer",
        join_timeout=1,
    )

    # start a streaming connection to the server that passes
    # server responses to the writer in a callback thread
    with stream(loader, client, writer):
        for i, witness_fname in enumerate(crawler):
            loader.in_q.put(witness_fname)
            time.sleep(1e-5)

            # always keep one extra frame in `loader.in_q`
            # so that it doesn't keep hitting stop iterations
            # when it gets to the end of a frame
            if i == 0:
                continue

            # submit all the inference requests up front
            for _ in range(int(inference_sampling_rate)):
                package = loader.get_package()

                client.in_q.put(package)
                time.sleep(1e-5)
                package = client.get_package()
                client.process(*package)

                # sleep to keep from overloading the server
                time.sleep(1 / inference_rate)

            # check to see if the writer produced any new
            # cleaned frames. Move on if not
            try:
                result = writer.out_q.get_nowait()
            except Empty:
                continue
            else:
                if isinstance(result, str):
                    raise RuntimeError(result)
                fname, latency = result

            logging.info(
                f"Wrote cleaned frame {fname} with latency {latency}s"
            )


if __name__ == "__main__":
    main()
