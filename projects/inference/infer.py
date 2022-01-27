import logging
import os
from typing import List, Union

import numpy as np
import tritonclient.grpc as triton
from gwpy.timeseries import TimeSeries, TimeSeriesDict
from hermes.typeo import typeo

import deepclean.infer.pseudo as infer
from deepclean.logging import configure_logging
from deepclean.signal import Pipeline


@typeo
def main(
    url: str,
    model_name: str,
    train_directory: str,
    witness_data_dir: str,
    strain_data_dir: str,
    channels: Union[str, List[str]],
    kernel_length: float,
    stride_length: float,
    sample_rate: float,
    max_latency: float,
    filter_memory: float,
    filter_lead_time: float,
    sequence_id: int = 1001,
    verbose: bool = False,
):
    configure_logging(os.path.join(train_directory, "infer.log"), verbose)

    client = triton.InferenceServerClient(url)
    if not client.is_server_live():
        raise RuntimeError(f"No server running at url {url}")
    elif not client.is_model_ready(model_name):
        raise RuntimeError(f"Model {model_name} not ready for inference")

    preprocessor = Pipeline.load(
        os.path.join(train_directory, "witness_pipeline.pkl")
    )
    postprocessor = Pipeline.load(
        os.path.join(train_directory, "strain_pipeline.pkl")
    )

    witness_fnames = sorted(os.listdir(witness_data_dir))
    strain_fnames = sorted(os.listdir(strain_data_dir))

    strains = np.array([])
    request_id = 0
    remainder = None

    logging.info("Beginning inference request submissions")
    with infer.begin_inference(
        client, model_name, max_latency, stride_length
    ) as (input, callback):
        for strain_fname, witness_fname in zip(strain_fnames, witness_fnames):
            logging.debug(
                f"Reading frame files {strain_fname} and {witness_fname}"
            )

            X = TimeSeriesDict.read(witness_fname, channels[1:])
            X = X.resample(sample_rate)
            X = np.stack([X[i].value for i in sorted(channels[1:])])
            X = preprocessor(X)
            if remainder is not None:
                X = np.conatenate([remainder, X], axis=-1)

            y = TimeSeries.read(strain_fname, channels[0])
            y = y.resample(sample_rate)
            strains = np.append(strains, y)

            remainder, request_id = infer.submit_for_inference(
                client=client,
                input=input,
                X=X,
                stride=int(stride_length * sample_rate),
                initial_request_id=request_id,
                sequence_id=sequence_id,
                model_name=model_name,
                sequence_end=False,  # TODO: best way to do this?
            )

            if callback.error is not None:
                raise callback.error

    logging.info("Producing cleaned frames from inference outputs")
    cleaned_frames = infer.online_postprocess(
        callback.predictions,
        strains,
        frame_length=1,  # TODO: best way to do this?
        postprocessor=postprocessor,
        filter_memory=filter_memory,
        filter_lead_time=filter_lead_time,
        sample_rate=sample_rate,
    )

    write_dir = os.path.join(train_directory, "cleaned")
    os.makedirs(write_dir, exist_ok=True)
    logging.info(f"Writing cleaned frames to '{write_dir}'")
    for fname, frame in zip(strain_fnames, cleaned_frames):
        fname = os.path.basename(fname)
        fname = os.path.join(write_dir, fname)

        ts = TimeSeries(frame, channel=channels[0])
        ts.write(fname)
        logging.debug(f"Wrote frame file {fname}")


if __name__ == "__main__":
    main()
