import logging
import time
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
):
    configure_logging(log_file, verbose)
    logger = logging.getLogger()

    channels = get_channels(channels)

    t0 = 0 if start_first else -1
    crawler = FrameCrawler(witness_data_dir, t0, timeout)
    loader = FrameLoader(
        inference_sampling_rate=inference_sampling_rate,
        sample_rate=sample_rate,
        channels=channels[1:],
        sequence_id=sequence_id,
        name="loader",
        rate=inference_rate,
        join_timeout=1,
    )
    loader.logger = logger

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
    writer.in_q = client.out_q
    writer.logger = logger

    time.sleep(1)
    try:
        client.client.start_stream(callback=client.callback)
        for i, witness_fname in enumerate(crawler):
            loader.in_q.put(witness_fname)
            if i == 0:
                continue

            for _ in range(int(inference_sampling_rate)):
                package = loader.get_package()

                client.in_q.put(package)
                package = client.get_package()
                client.process(*package)

                response = writer.get_package()
                writer.process(response)

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
            finally:
                if start_first:
                    time.sleep(inference_sampling_rate / inference_rate)
    finally:
        client.client.close()
        for p in [loader, client, writer]:
            for q in [p.in_q, p.out_q]:
                while True:
                    try:
                        q.get_nowait()
                    except (Empty, ValueError):
                        break

                try:
                    q.close()
                except ValueError:
                    pass
                q.join_thread()


if __name__ == "__main__":
    main()
