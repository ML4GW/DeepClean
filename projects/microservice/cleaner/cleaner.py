import logging
import time
from pathlib import Path
from queue import Empty
from typing import Iterable, Optional, Union

from hermes.stillwater import InferenceClient
from hermes.stillwater.utils import ExceptionWrapper
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
    channels = get_channels(channels)

    crawler = FrameCrawler(
        witness_data_dir, strain_data_dir, start_first, timeout
    )
    loader = FrameLoader(
        inference_sampling_rate=inference_sampling_rate,
        sample_rate=sample_rate,
        channels=channels,
        sequence_id=sequence_id,
        name="loader",
        rate=inference_rate,
        join_timeout=1,
    )

    client = InferenceClient(
        url=url,
        model_name=model_name,
        model_version=model_version,
        profile=False,
        name="client",
        join_timeout=1,
    )

    postprocessor = BandpassFilter(freq_low, freq_high, sample_rate)
    metadata = client.client.get_model_metadata(client.model_name)

    if max_latency is None:
        aggregation_steps = 0
    else:
        aggregation_steps = int(max_latency * inference_sampling_rate)
    writer = FrameWriter(
        write_dir,
        channel_name=channels[0] + "-CLEANED",
        inference_sampling_rate=inference_sampling_rate,
        sample_rate=sample_rate,
        strain_q=loader.strain_q,
        postprocessor=postprocessor,
        memory=memory,
        look_ahead=look_ahead,
        aggregation_steps=aggregation_steps,
        output_name=metadata.outputs[0].name,
        name="writer",
        join_timeout=1,
    )
    time.sleep(1)

    pipeline = loader >> client >> writer
    with pipeline:
        for witness_fname, strain_fname in crawler:
            loader.in_q.put((witness_fname, strain_fname))

            try:
                result = writer.out_q.get_nowait()
            except Empty:
                continue
            else:
                if isinstance(result, ExceptionWrapper):
                    result.reraise()

                fname, latency = result
                logging.info(
                    "Wrote cleaned frame {} with latency {}s".format(
                        fname, latency
                    )
                )
            finally:
                if start_first:
                    time.sleep(0.9)

        loader.in_q.put(StopIteration)


if __name__ == "__main__":
    main()
