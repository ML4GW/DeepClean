import logging
import os
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
from gwpy.timeseries import TimeSeries, TimeSeriesDict
from hermes.typeo import typeo
from tritonclient import grpc as triton

import deepclean.infer.pseudo as infer
from deepclean.logging import configure_logging
from deepclean.serve import serve
from deepclean.signal.filter import FREQUENCY, BandpassFilter


def write_frames(frames, write_dir: Path, fnames: str, channel: str):
    write_dir.mkdir(parents=True, exist_ok=True)
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
    model_repo_dir: str,
    model_name: str,
    output_directory: Path,
    witness_data_dir: Path,
    strain_data_dir: Path,
    channels: Union[str, List[str]],
    kernel_length: float,
    stride_length: float,
    sample_rate: float,
    max_latency: float,
    memory: float,
    look_ahead: float,
    freq_low: FREQUENCY,
    freq_high: FREQUENCY,
    sequence_id: int = 1001,
    verbose: bool = False,
    gpus: Optional[List[int]] = None,
    max_frames: Optional[int] = None,
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
        witness_data_dir:
            A directory containing one-second-long gravitational
            wave frame files corresponding to witness data as
            inputs to DeepClean. Files should be named identically
            except for their end which should take the form
            `<GPS timestamp>_<length of frame>.gwf`.
        strain_data_dir:
            A directory containing one-second-long gravitational
            wave frame files corresponding to the strain data
            to be cleaned. The same rules about naming conventions
            apply as those outlined for the files in `witness_data_dir`,
            with the added stipulation that each timestamp should have
            a matching file in `witness_data_dir`.
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
        max_frames:
            The maximum number of files from `witness_data_dir` and
            `strain_data_dir` to clean.
    """

    # load the channels from a file if we specified one
    if isinstance(channels, str) or len(channels) == 1:
        if isinstance(channels, str):
            channels = [channels]

        with open(channels[0], "r") as f:
            channels = [i for i in f.read().splitlines() if i]

    configure_logging(output_directory / "infer.log", verbose)

    # launch a singularity container in a separate thread
    # that runs the server, captures its logs, and waits until
    # the server comes online before entering the context
    with serve(
        model_repo_dir,
        gpus=gpus,
        log_file=output_directory / "server.log",
        wait=True,
    ):
        # connect to the server at the given url
        client = triton.InferenceServerClient(url)

        # TODO: some checks to make sure that the filenames match,
        # or just more intelligent logic to iterate through these.
        # How many assumptions about the naming conventions do
        # we want to make?
        witness_fnames = sorted(os.listdir(witness_data_dir))
        strain_fnames = sorted(os.listdir(strain_data_dir))
        N = min(max_frames or np.inf, len(witness_fnames))

        strains = np.array([])
        request_id = 0
        remainder = None

        # do actual inference in a context that maintains a
        # connection to the server to make sure the snapshot
        # state is updated sequentially and responses are handled
        # in a separate thread (managed by the `callback` returned here,
        # which will maintain the model's predictions in-memory in a
        # `predictions` attribute)
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
                X = X.astype("float32")

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
                    callback=callback,
                )

                # check to see if the server raised an error
                # that got processed by our callback
                if callback.error is not None:
                    raise RuntimeError(callback.error)

        # if we did some server-side aggregation, we need to get
        # rid of the first few update steps since they technically
        # correspond to times _before_ the input data began
        update_steps = max_latency // stride_length
        stride_size = sample_rate * stride_length
        throw_away = int(update_steps * stride_size)
        postprocessor = BandpassFilter(freq_low, freq_high, sample_rate)

        # now clean the processed timeseries in an
        # online fashion and break into frames
        logging.info("Producing cleaned frames from inference outputs")
        cleaned_frames = infer.online_postprocess(
            callback.predictions[throw_away:],
            strains[:-throw_away],
            frame_length=1,  # TODO: best way to do this?
            postprocessor=postprocessor,
            memory=memory,
            look_ahead=look_ahead,
            sample_rate=sample_rate,
        )

        # now write these frames to the directory where
        # all our training outputs go. TODO: should this
        # be an optional param that defaults to this value?
        write_dir = output_directory / "cleaned"
        logging.info(f"Writing cleaned frames to '{write_dir}'")
        channel_name = channels[0] + "-CLEANED"
        write_frames(cleaned_frames, write_dir, strain_fnames, channel_name)


if __name__ == "__main__":
    main()
