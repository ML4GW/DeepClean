import time
from pathlib import Path

from gwpy.timeseries import TimeSeriesDict
from microservice.deployment import DataStream, Deployment, ExportClient
from microservice.frames import DataProducts, FrameCrawler, load_frame

from deepclean.logging import logger
from deepclean.utils.channels import ChannelList, get_channels
from deepclean.utils.filtering import Frequency, normalize_frequencies
from typeo import scriptify


@scriptify
def main(
    # paths
    run_directory: Path,
    data_directory: Path,
    data_field: str,
    export_endpoint: str,
    # data/analysis args
    channels: ChannelList,
    sample_rate: float,
    fftlength: float,
    asd_segment_length: float,
    monitor_period: float,
    freq_low: Frequency,
    freq_high: Frequency,
    # storage args
    max_files: int,
    write_cadence: int,
    # misc args
    verbose: bool = False,
):
    deployment = Deployment(run_directory)
    log_file = deployment.log_directory / "monitor.log"
    logger.set_logger("DeepClean infer", log_file, verbose)
    export_client = ExportClient(export_endpoint)

    channels = get_channels(channels)
    strain_channel = channels[0]
    data_products = DataProducts(strain_channel)

    freq_low, freq_high = normalize_frequencies(freq_low, freq_high)
    freq_low, freq_high = freq_low[0], freq_high[0]

    # wait for inference service to begin writing frames
    start_time = time.time()
    while not list(deployment.frame_directory.iterdir()):
        time.sleep(1)
        div, mod = divmod(time.time() - start_time, 10)
        if not mod:
            logger.info(
                "Waiting for first frame to be written, "
                "{}s elapsed".format(div)
            )

    # now start iterating through written frames and
    # 1. Periodically move them to larger files in cold storage
    # 2. Remove files to keep frame directory at fixed max length
    # 3. Monitor frames cleaned by the canary model to ensure that
    #     it's in spec before moving it into production
    stream = DataStream(data_directory, data_field)
    files = []
    buffer, t0 = TimeSeriesDict(), None

    latest_version = 1
    validating = False
    in_spec_for = 0

    for fname in FrameCrawler(deployment.frame_directory, t0=None):
        ifo, field, timestamp, *_ = fname.stem.split("-")
        timestamp = int(timestamp)
        if t0 is None:
            t0 = timestamp

        # grab all of the data products from the current
        # frame as well as its associate raw data
        strain_fname = stream.hoft / fname.name
        frame = load_frame(fname, data_products.channels, sample_rate)
        raw = load_frame(strain_fname, strain_channel, sample_rate)

        frame[strain_channel] = raw
        buffer.append(frame)

        # see if we've aggregated enough data to move
        # into cold storage in our output directory
        if (timestamp - t0) == write_cadence:
            # slice out the current segment for writing, ignoring
            # any data from the past that we might have had to hold
            # on to for canary validation
            output = buffer.crop(t0, timestamp)
            write_fname = f"{ifo}-{field}-{output.t0}-{write_cadence}.h5"
            write_path = deployment.storage_directory / write_fname

            logger.info(
                "Accumulated {}s of frame data, writing to {}".format(
                    write_cadence, write_path
                )
            )
            output.write(write_path)

            # now slough off the data from the frame we just
            # wrote, saving just enough to be able to keep
            # doing canary model validation. Increment t0 to mathc
            start = timestamp - asd_segment_length
            buffer = buffer.crop(start, timestamp)
            t0 = timestamp

        files.append(fname)
        if len(files) > max_files:
            removed = files.pop(0)
            logger.info(f"Removing frame file {removed}")
            removed.unlink()

        # if we don't have enough data to validate the ASD, move on
        if buffer.duration < asd_segment_length:
            continue

        # if we're not currently validating, see if there is a
        # newer version of the model that needs validation
        if not validating:
            version = export_client.latest_version()
            if version > latest_version:
                latest_version = version
                validating = True
                in_spec_for = 0
            else:
                # if we're still on the same version, move on
                continue

        # crop out the last asd_segment_length seconds worth
        # of data from both the raw and canary-cleaned timeseries.
        # Compute their ASDs over the relevant frequency range
        start = timestamp - asd_segment_length

        raw = buffer[strain_channel].crop(start, timestamp)
        raw = raw.asd(fftlength, method="median", window="hann")
        raw = raw.crop(freq_low, freq_high)

        clean = buffer[data_products.canary].crop(start, timestamp)
        clean = clean.asd(fftlength, method="median", window="hann")
        clean = clean.crop(freq_low, freq_high)

        # measure the ratio of these ASDs and increment
        # our counter if it's in spec.
        # TODO: write this to some sort of Prometheus
        # dashboard or something
        asdr = clean / raw
        if asdr.mean() <= 1:
            in_spec_for += 1
        else:
            in_spec_for = 0

        # if we've been in spec for the right period,
        # then increment the production version of
        # DeepClean and reset our validation counter
        if in_spec_for >= monitor_period:
            export_client.increment()
            validating = False
            in_spec_for = 0
