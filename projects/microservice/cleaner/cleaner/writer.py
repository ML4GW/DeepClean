import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

from gwpy.timeseries import TimeSeries, TimeSeriesDict
from microservice.deployment import ExportClient
from microservice.frames import DataProducts
from scipy.signal.windows import hann

from deepclean.logging import logger
from deepclean.utils.filtering import Frequency, normalize_frequencies


@dataclass
class ASDRMonitor:
    buffer_length: float
    freq_low: Frequency
    freq_high: Frequency
    fftlength: float = 2
    overlap: Optional[float] = None

    def __post_init__(self):
        self.raw, self.clean, self.window = None, None, None
        self.freq_low, self.freq_high = normalize_frequencies(
            self.freq_low, self.freq_high
        )
        self.freq_low = self.freq_low[0]
        self.freq_high = self.freq_high[0]

        # always start out of spec until we can verify
        # that our cleaned data is good
        self.in_spec = False
        self.logger = logger.get_logger("DeepClean online monitor")

    def get_asd(self, ts: TimeSeries):
        asd = ts.asd(
            self.fftlength,
            overlap=self.overlap,
            window="hann",
            method="median",
        )
        return asd.crop(self.freq_low, self.freq_high)

    def __call__(self, raw: TimeSeries, noise: TimeSeries) -> TimeSeries:
        clean = raw - noise
        if self.raw is None:
            # special short-circuit for the first data we get
            self.raw = raw
            self.clean = clean

            # build a Tukey window for tapering
            # the cleaned data stream in and out
            sample_rate = raw.sample_rate.value
            size = int(sample_rate * 2)
            cutoff = int(sample_rate)
            self.window = hann(size)[-cutoff:]
            return None, None

        self.raw = self.raw.append(raw, inplace=False)
        self.clean = self.clean.append(clean, inplace=False)

        # crop off the second to last frame from
        # each of our buffers to potentially write
        # depending on if quality check are passed
        t0, tf = self.raw.span
        raw = self.raw.crop(tf - 2, tf - 1)
        clean = self.clean.crop(tf - 2, tf - 1)

        # if we don't have enough data to measure the ASDR
        # yet, then just return the buffered raw data
        duration = self.raw.duration.value
        if duration < self.buffer_length:
            self.logger.info(
                "Buffer only {:0.0f}s long, not enough to "
                "measure ASDR so only producing raw data".format(duration)
            )
            return raw, raw

        # check the average ASDR over the relevant
        # frequency bins. TODO: do we want to do a
        # more in-depth analysis than this?
        raw_asd = self.get_asd(self.raw)
        clean_asd = self.get_asd(self.clean)
        asdr = (clean_asd / raw_asd).value.mean()

        self.logger.info(
            "Mean ASDR value over GPS times {}-{} "
            "and frequency range {}-{}Hz is currently {}".format(
                t0, tf, self.freq_low, self.freq_high, asdr
            )
        )

        if asdr > 1 and self.in_spec:
            # we've gone out of spec, so reconstruct the
            # noise from the buffered frame and taper it
            # before re-subtracting it, then set the
            # in_spec flag so that future frames will
            # return the raw data until we've fixed the issue
            self.logger.warning(
                "Mean ASDR value in frequency range {}-{}Hz "
                "is now {}, tapering noise predictions to 0 "
                "until ASDR returns below 1".format(
                    self.freq_low, self.freq_high, asdr
                )
            )

            self.in_spec = False
            noise = (raw - clean) * self.window
            frame = raw - noise
        elif asdr > 1 and not self.in_spec:
            self.logger.debug(
                "Mean ASDR is still out of spec, writing raw strain"
            )
            # we're currently out of spec and the most recent
            # frame hasn't changed that, so continue to
            # return the uncleaned data
            frame = raw
        elif asdr < 1 and not self.in_spec:
            # we were out of spec, but the most recent frame
            # brought us back in, so taper in the cleaned
            # data again starting with the buffered frame
            self.logger.info(
                "Mean ASDR value in frequency range {}-{}Hz "
                "is now {}, tapering noise subtraction back "
                "into production data stream".format(
                    self.freq_low, self.freq_high, asdr
                )
            )

            self.in_spec = True
            noise = (raw - clean) * self.window[::-1]
            frame = raw - noise
        else:
            self.logger.debug(
                "Mean ASDR is still in spec, writing cleaned strain"
            )
            frame = clean

        # finally, if we've reached our buffer length,
        # crop off the earliest part of the buffer
        if self.raw.duration.value >= self.buffer_length:
            self.raw = self.raw.crop(t0 + 1, tf)
            self.clean = self.clean.crop(t0 + 1, tf)

        return frame, clean


@dataclass
class Writer:
    write_dir: Path
    strain_generator: Iterator
    monitor: ASDRMonitor
    export_endpoint: str

    def __post_init__(self):
        self.write_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logger.get_logger("DeepClean writer")
        self.export_client = ExportClient(self.export_endpoint)
        self.strain_buffer = None
        self.canary_buffer = None
        self.timestamp_buffer = None

    def __call__(self, **predictions):
        strain, fname = next(self.strain_generator)

        # check which version of the model cleaned this frame
        # TODO: incorporate this as metadata somehow
        version = self.export_client.get_production_version()
        self.logger.info(
            "Cleaning strain from file {} using predictions "
            "made by DeepClean version {}".format(fname, version)
        )

        # extract some metadata about the frame we just got
        sample_rate = strain.sample_rate.value
        ifo, field, timestamp, dur = fname.stem.split("-")

        # the timestamp of the frame we _write_ will be
        # one second behind the timestamp of this frame.
        # Record the actual time the frame got written
        # for the purposes of measuring latency
        timestamp = int(timestamp)
        t0 = timestamp - 1
        file_timestamp = os.path.getmtime(fname)

        # our frames will contain three channels as their
        # data products, each one appending something
        # different to the name of the strain channel
        data_products = DataProducts(strain.channel.name)
        tsd = TimeSeriesDict()

        for key, noise in predictions.items():
            # parse out "canary" or "production" from the
            # name of the prediction tensor
            model = key.split("-")[1]
            if model == "production" and version == 1:
                # if the production model is still the
                # randomly initialized version of DeepClean,
                # then ignore the noise prediction altogether
                # and just write the plain strain data
                self.logger.debug(
                    "Production model still randomly initialized, "
                    "skipping production clean"
                )
                if self.strain_buffer is None:
                    continue

                tsd[data_products.out_dq] = TimeSeries(
                    strain.value,
                    t0=t0,
                    sample_rate=sample_rate,
                    channel=data_products.out_dq,
                )
                tsd[data_products.production] = TimeSeries(
                    strain.value,
                    t0=t0,
                    sample_rate=sample_rate,
                    channel=data_products.production,
                )
            elif model == "production":
                # the production model needs to be cleaned
                # specially to ensure that the ASDR matches
                # our expectations, otherwise we'll just return
                # the raw frame
                noise = TimeSeries(
                    noise, t0=timestamp, sample_rate=sample_rate
                )
                dq, clean = self.monitor(strain, noise)
                if clean is None:
                    continue

                tsd[data_products.out_dq] = TimeSeries(
                    dq.value,
                    t0=t0,
                    sample_rate=sample_rate,
                    channel=data_products.out_dq,
                )
                tsd[data_products.production] = TimeSeries(
                    clean.value,
                    t0=t0,
                    sample_rate=sample_rate,
                    channel=data_products.production,
                )
            elif model == "canary":
                # for the canary model, always clean normally
                clean = strain - noise
                if self.canary_buffer is not None:
                    tsd[data_products.canary] = TimeSeries(
                        self.canary_buffer.value,
                        t0=t0,
                        sample_rate=sample_rate,
                        channel=data_products.canary,
                    )
                self.canary_buffer = clean
            else:
                raise ValueError("Unrecognized model {model}")

        # short circuit if we didn't add anything
        # to our timeseries dict
        self.strain_buffer = strain
        if not tsd:
            # record the timestamp of this first file
            # to compute our write latency later
            self.timestamp_buffer = file_timestamp
            self.logger.info("Skipping clean of first file")
            return None

        write_name = f"{ifo}-{field}-{t0}-{dur}.gwf"
        write_path = self.write_dir / write_name
        tsd.write(write_path)
        self.logger.info(f"Finished cleaning of file {write_path}")

        write_time = time.time()
        latency = write_time - self.timestamp_buffer
        self.timestamp_buffer = file_timestamp

        return write_path, latency
