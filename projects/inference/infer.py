import logging
import os
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import tritonclient.grpc as triton
from gwpy.timeseries import TimeSeries, TimeSeriesDict
from hermes.typeo import typeo

import deepclean.infer.pseudo as infer
from deepclean.logging import configure_logging
from deepclean.signal import Pipeline


def write_frames(frames, write_dir: Path, fnames: str, channel: str):
    os.makedirs(write_dir, exist_ok=True)
    for fname, frame in zip(fnames, frames):
        # use the filename and channel name from the
        # strain data for the cleaned frame. TODO:
        # should we introduce "DC" somewhere?
        fname = write_dir / fname
        ts = TimeSeries(frame, channel=channel)
        ts.write(fname)
        logging.debug(f"Wrote frame file '{fname}'")


@typeo
def main(
    url: str,
    model_name: str,
    train_directory: Path,
    witness_data_dir: Path,
    strain_data_dir: Path,
    channels: Union[str, List[str]],
    kernel_length: float,
    stride_length: float,
    sample_rate: float,
    max_latency: float,
    filter_memory: float,
    filter_lead_time: float,
    sequence_id: int = 1001,
    verbose: bool = False,
    max_frames: Optional[int] = None,
):
    configure_logging(train_directory / "infer.log", verbose)

    # connect to the server at the given
    # url and make sure it's running and
    # has the desired model online
    client = triton.InferenceServerClient(url)
    if not client.is_server_live():
        raise RuntimeError(f"No server running at url {url}")
    elif not client.is_model_ready(model_name):
        raise RuntimeError(f"Model {model_name} not ready for inference")

    # load in our pre- and postprocessing objects
    preprocessor = Pipeline.load(train_directory / "witness_pipeline.pkl")
    postprocessor = Pipeline.load(train_directory / "strain_pipeline.pkl")

    witness_fnames = sorted(os.listdir(witness_data_dir))
    strain_fnames = sorted(os.listdir(strain_data_dir))
    N = min(max_frames or np.inf, len(witness_fnames))

    strains = np.array([])
    request_id = 0
    remainder = None

    logging.info("Beginning inference request submissions")
    with infer.begin_inference(client, model_name) as (input, callback):
        for i in range(N):
            # grab the ith filenames for processing
            witness_fname = witness_data_dir / witness_fnames[i]
            strain_fname = strain_data_dir / strain_fnames[i]
            logging.debug(
                "Reading frames '{}' and '{}'".format(
                    strain_fname, witness_fname
                )
            )

            # load in and preprocess the witnesses
            X = TimeSeriesDict.read(witness_fname, channels[1:])
            X = X.resample(sample_rate)
            X = np.stack([X[i].value for i in sorted(channels[1:])])
            X = preprocessor(X).astype("float32")

            # tack on any leftover witnesses from the last
            # frame that weren't sufficiently long to make
            # an inference request
            if remainder is not None:
                X = np.conatenate([remainder, X], axis=-1)

            # load in the corresponding strain data
            # and tack it on to our running array
            y = TimeSeries.read(strain_fname, channels[0])
            y = y.resample(sample_rate)
            strains = np.append(strains, y)

            # make a series of inference requests using the
            # the witnesses. The outputs will be tacked on
            # to the `predictions` attribute of our `callback`
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

            # check to see if the server raised an error
            # that got processed by our callback
            if callback.error is not None:
                raise callback.error

    # if we did some server-side aggregation, we need to get
    # rid of the first few update steps since they technically
    # correspond to times _before_ the input data began
    update_steps = max_latency // stride_length
    stride_size = sample_rate * stride_length
    throw_away = int(update_steps * stride_size)

    # now clean the processed timeseries in an
    # online fashion and break into frames
    logging.info("Producing cleaned frames from inference outputs")
    preds, posts, cleaned_frames = infer.online_postprocess(
        callback.predictions[throw_away:],
        strains[:-throw_away],
        frame_length=1,  # TODO: best way to do this?
        postprocessor=postprocessor,
        filter_memory=filter_memory,
        filter_lead_time=filter_lead_time,
        sample_rate=sample_rate,
    )

    # now write these frames to the directory where
    # all our training outputs go. TODO: should this
    # be an optional param that defaults to this value?
    write_dir = train_directory / "cleaned"
    logging.info(f"Writing cleaned frames to '{write_dir}'")
    write_frames(cleaned_frames, write_dir, strain_fnames, channels[0])

    write_frames(
        preds, train_directory / "predictions", strain_fnames, channels[0]
    )
    write_frames(
        posts, train_directory / "postprocessed", strain_fnames, channels[0]
    )


if __name__ == "__main__":
    main()
