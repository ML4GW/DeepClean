import logging
from collections.abc import Iterable
from pathlib import Path
from queue import Empty, Queue
from typing import Optional, Union

from hermes.stillwater import InferenceClient
from hermes.stillwater.utils import ExceptionWrapper
from hermes.typeo import typeo

from deepclean.infer.asynchronous import FrameWriter
from deepclean.infer.frame_crawler import FrameCrawler
from deepclean.logging import configure_logging
from deepclean.signal.filter import BandpassFilter


@typeo
def main(
    witness_data_dir: Path,
    strain_data_dir: Path,
    write_dir: Path,
    sample_rate: float,
    kernel_length: float,
    inference_sampling_rate: float,
    inference_rate: float,
    channels: Union[str, Iterable[str]],
    freq_low: Union[float, Iterable[float]],
    freq_high: Union[float, Iterable[float]],
    url: str,
    model_name: str,
    model_version: int = -1,
    sequence_id: int = 1001,
    max_latency: Optional[float] = None,
    start_first: bool = False,
    timeout: Optional[float] = None,
    log_file: Optional[str] = None,
    memory: float = 10,
    look_ahead: float = 0.5,
    verbose: bool = False,
):
    configure_logging(log_file, verbose)

    crawler = FrameCrawler(
        witness_data_dir, strain_data_dir, start_first, timeout
    )

    client = InferenceClient(
        url=url,
        model_name=model_name,
        model_version=model_version,
        profile=False,
        name="client",
        rate=inference_rate,
    )
    loader = ""  # TODO

    postprocessor = BandpassFilter(freq_low, freq_high)
    metadata = client.client.get_model_metadata(client.model_name)
    writer = FrameWriter(
        write_dir,
        channel_name=channels[0] + "-CLEANED",
        inference_sampling_rate=inference_sampling_rate,
        sample_rate=sample_rate,
        strain_q=Queue(),
        postprocessor=postprocessor,
        memory=memory,
        look_ahead=look_ahead,
        aggregation_steps=int(max_latency * inference_sampling_rate),
        output_name=metadata.outputs[0].name,
        name="writer",
    )

    pipeline = client >> writer
    stride = int(sample_rate // inference_sampling_rate)
    with pipeline:
        for witness_fname, strain_fname in crawler:
            witness, strain = loader(witness_fname, strain_fname)
            writer.strain_q.put(((witness_fname, strain_fname), strain))

            for i in range(inference_sampling_rate):
                slc = slice(i * stride, (i + 1) * stride)
                client.in_q.put(witness[:, slc])

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


if __name__ == "__main__":
    main()
