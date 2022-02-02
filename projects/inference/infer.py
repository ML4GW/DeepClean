import logging
import os
from typing import List, Optional, Union

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
    max_frames: Optional[int] = None
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
    N = min(max_frames or np.inf, len(witness_fnames))

    strains = np.array([])
    request_id = 0
    remainder = None

    logging.info("Beginning inference request submissions")
    infer_ctx = infer.begin_inference(client, model_name)
    with infer_ctx as (input, callback):
        for i in range(N):
            witness_fname = os.path.join(witness_data_dir, witness_fnames[i])
            strain_fname = os.path.join(strain_data_dir, strain_fnames[i])
            logging.debug(
                f"Reading frame files '{strain_fname}' and '{witness_fname}'"
            )

            X = TimeSeriesDict.read(witness_fname, channels[1:])
            X = X.resample(sample_rate)
            X = np.stack([X[i].value for i in sorted(channels[1:])])
            X = preprocessor(X).astype("float32")
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
    throw_away = int(max_latency // stride_length * sample_rate * stride_length)
    print(throw_away)
    cleaned_frames = infer.online_postprocess(
        callback.predictions[throw_away:],
        strains[:-throw_away],
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
