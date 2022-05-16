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
        response = writer.get_package()
        writer.process(response)

    try:
        client.client.start_stream(callback=callback)
        yield client
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
    configure_logging(log_file, verbose)
    logger = logging.getLogger()

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

    # for this and all other processes here, manually
    # set the logger since this normally gets set during
    # `PipelineProcess.run`
    loader.logger = logger

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
    client.logger = logger

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
    writer.logger = logger

    # pipe the output of the client process to the input
    # of the writer process
    writer.in_q = client.out_q

    time.sleep(1)
    with stream(loader, client, writer):
        for i, witness_fname in enumerate(crawler):
            loader.in_q.put(witness_fname)

            # always keep one extra frame in `loader.in_q`
            # so that it doesn't keep hitting stop iterations
            # when it gets to the end of a frame
            if i == 0:
                continue

            # submit all the inference requests up front
            for _ in range(int(inference_sampling_rate)):
                package = loader.get_package()

                client.in_q.put(package)
                package = client.get_package()
                client.process(*package)
                time.sleep(1 / inference_rate)

            # check to see if the writer produced any new
            # cleaned frames. Move on if not
            try:
                fname, latency = writer.out_q.get_nowait()
            except Empty:
                continue
            else:
                logging.info(
                    "Wrote cleaned frame {} with latency {}s".format(
                        fname, latency
                    )
                )


if __name__ == "__main__":
    main()
