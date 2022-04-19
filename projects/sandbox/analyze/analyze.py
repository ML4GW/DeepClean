import os
import re
from collections.abc import Iterable
from pathlib import Path
from typing import List, Optional

import numpy as np
from bokeh.io import save
from bokeh.layouts import column, row
from bokeh.models import (
    BoxZoomTool,
    ColumnDataSource,
    Div,
    HoverTool,
    Panel,
    PreText,
    Tabs,
)
from bokeh.palettes import Colorblind8 as palette
from bokeh.plotting import figure
from gwpy.timeseries import TimeSeries
from hermes.typeo import typeo
from scipy.signal import welch

from deepclean.gwftools.channels import ChannelList, get_channels


def build_timeseries(
    fnames: Iterable[str], channel: str, sample_rate: float
) -> np.ndarray:
    """
    Read in the indicated channel from a sequence of .gwf filenames,
    resample them to the indicated sample rate and concatenate them
    into a single timeseries.
    """

    # TODO: too many assumptions about length of frames
    h = np.zeros((int(len(fnames) * sample_rate),))
    for i, fname in enumerate(fnames):
        ts = TimeSeries.read(fname, channel=channel).resample(sample_rate)
        h[i * int(sample_rate) : (i + 1) * int(sample_rate)] = ts.value
    return h


def make_asd(
    data: np.ndarray,
    sample_rate: float,
    fftlength: float,
    overlap: Optional[float] = None,
) -> np.ndarray:
    """
    Compute the ASD of a given timeseries and return it along
    with its corresponding frequency bin values
    """

    nperseg = int(fftlength * sample_rate)
    overlap = overlap or fftlength / 2
    noverlap = int(overlap * sample_rate)

    f, psd = welch(data, fs=sample_rate, nperseg=nperseg, noverlap=noverlap)
    return f, np.sqrt(psd)


def get_training_curves(output_directory: Path):
    """
    Read the training logs from the indicated output directory
    and use them to parse the training and validation loss values
    at each epoch. Plot these curves to a Bokeh `Figure` and return it.
    """

    with open(output_directory / "train.log", "r") as f:
        train_log = f.read()

    epoch_re = re.compile("(?<==== Epoch )[0-9]{1,4}")
    train_loss_re = re.compile(r"(?<=Train Loss: )[0-9.e\-+]+")
    valid_loss_re = re.compile(r"(?<=Valid Loss: )[0-9.e\-+]+")

    source = ColumnDataSource(
        dict(
            epoch=list(map(int, epoch_re.findall(train_log))),
            train=list(map(float, train_loss_re.findall(train_log))),
            valid=list(map(float, valid_loss_re.findall(train_log))),
        )
    )

    p = figure(
        height=300,
        width=600,
        # sizing_mode="scale_width",
        title="Training curves",
        x_axis_label="Epoch",
        y_axis_label="ASDR",
        tools="reset,box_zoom",
    )

    r = p.line(
        "epoch",
        "train",
        line_width=2.3,
        line_color=palette[-1],
        line_alpha=0.8,
        legend_label="Train Loss",
        source=source,
    )
    p.line(
        "epoch",
        "valid",
        line_width=2.3,
        line_color=palette[-2],
        line_alpha=0.8,
        legend_label="Valid Loss",
        source=source,
    )

    p.add_tools(
        HoverTool(
            mode="vline",
            line_policy="nearest",
            point_policy="snap_to_data",
            renderers=[r],
            tooltips=[
                ("Epoch", "@epoch"),
                ("Train ASDR", "@train"),
                ("Valid ASDR", "@valid"),
            ],
        )
    )
    return p


def get_logs_box(output_directory: Path):
    """
    Creating a Bokeh `Tabs` layout with panels for each
    of the log files contained in `output_directory`, as well
    as of the config used to generate this run of DeepClean.
    """

    # TODO: make more general, include pyproject as a separate
    # arg that defaults to this value, in case this is being
    # run standalone.
    with open(Path(__file__).parent / ".." / "pyproject.toml", "r") as f:
        text_box = PreText(
            text=f.read(),
            height=1,
            width=1,
            style={
                "overflow-y": "scroll",
                "height": "250px",
                "overflow-x": "scroll",
                "width": "600px",
            },
        )
    panels = [Panel(child=text_box, title="Config")]

    for fname in os.listdir(output_directory):
        if fname.endswith(".log"):
            with open(output_directory / fname, "r") as f:
                text_box = PreText(
                    text=f.read(),
                    height=1,
                    width=1,
                    style={
                        "overflow-y": "scroll",
                        "height": "250px",
                        "overflow-x": "scroll",
                        "width": "600px",
                    },
                )
            panel = Panel(child=text_box, title=fname.split(".")[0].title())
            panels.append(panel)

    return Tabs(tabs=panels)


def get_asdr_vs_time(
    clean_timeseries: np.ndarray,
    raw_timeseries: np.ndarray,
    window_length: float,
    window_step: float,
    fftlength: float,
    sample_rate: float,
    freq_low: Optional[float] = None,
    freq_high: Optional[float] = None,
    overlap: Optional[float] = None,
):
    window_size = int(window_length * sample_rate)
    step_size = int(window_step * sample_rate)
    num_asdrs = (len(clean_timeseries) - window_size) // step_size + 1

    asdrs = []
    for i in range(num_asdrs):
        clean_window = clean_timeseries[i * step_size : (i + 1) * step_size]
        raw_window = raw_timeseries[i * step_size : (i + 1) * step_size]

        freqs, clean_asd = make_asd(
            clean_window, sample_rate, fftlength, overlap
        )
        freqs, raw_asd = make_asd(raw_window, sample_rate, fftlength, overlap)

        if freq_low is not None:
            mask = (freq_low <= freqs) & (freqs < freq_high)
            clean_asd = clean_asd[mask]
            raw_asd = raw_asd[mask]
        asdrs.append((clean_asd / raw_asd).mean())
    return asdrs


def analyze_test_data(
    raw_data_dir: Path,
    clean_data_dir: Path,
    output_directory: Path,
    channels: List[str],
    sample_rate: float,
    fftlength: float,
    freq_low: Optional[float] = None,
    freq_high: Optional[float] = None,
    overlap: Optional[float] = None,
):
    """
    Build plots of the ASDs of the cleaned and uncleaned
    data, as well as a plot of the ratio of these ASDs
    over a target frequency band. Return these plots as a
    Bokeh `row` layout, and return the number of frames
    analyzed as well.
    """

    fnames = sorted(os.listdir(clean_data_dir))
    raw_fnames = [raw_data_dir / f for f in fnames]
    clean_fnames = [clean_data_dir / f for f in fnames]

    raw_data = build_timeseries(raw_fnames, channels[0], sample_rate)
    clean_data = build_timeseries(clean_fnames, channels[0], sample_rate)

    freqs, raw_asd = make_asd(raw_data, sample_rate, fftlength, overlap)
    freqs, clean_asd = make_asd(clean_data, sample_rate, fftlength, overlap)
    asd_source = ColumnDataSource(
        {"raw": raw_asd, "clean": clean_asd, "freqs": freqs}
    )

    asdr = clean_asd / raw_asd
    if freq_low is not None and freq_high is not None:
        mask = (freq_low <= freqs) & (freqs <= freq_high)
        freqs = freqs[mask]
        asdr = asdr[mask]
    elif freq_low is None or freq_high is None:
        # TODO: make more explicit
        raise ValueError(
            "freq_low and freq_high must both be either None or float"
        )
    asdr_source = ColumnDataSource({"asdr": asdr, "freqs": freqs})

    p_asd = figure(
        height=300,
        width=600,
        title="ASD of Raw and Clean Strains",
        y_axis_type="log",
        x_axis_label="Frequency [Hz]",
        y_axis_label="ASD [Hz⁻¹ᐟ²]",
        sizing_mode="scale_width",
        tools="reset",
    )

    for i, asd in enumerate(["raw", "clean"]):
        r = p_asd.line(
            x="freqs",
            y=asd,
            line_color=palette[i],
            line_width=2.3,
            line_alpha=0.8,
            legend_label=asd.title(),
            source=asd_source,
        )
    p_asd.add_tools(
        HoverTool(
            renderers=[r],
            tooltips=[
                ("Frequency", "@freqs Hz"),
                ("Power of raw strain", "@raw"),
                ("Power of clean strain", "@clean"),
            ],
            mode="vline",
        ),
        BoxZoomTool(dimensions="width"),
    )
    p_asd.legend.click_policy = "hide"

    p_asdr = figure(
        height=300,
        width=600,
        title="Ratio of clean strain to raw strain",
        x_axis_label="Frequency [Hz]",
        y_axis_label="Ratio [Clean / Raw]",
        sizing_mode="scale_width",
        tools="reset",
    )
    p_asdr.line(
        x="freqs",
        y="asdr",
        line_color=palette[2],
        line_width=2.3,
        line_alpha=0.8,
        source=asdr_source,
    )
    p_asdr.add_tools(
        HoverTool(
            renderers=[r],
            tooltips=[("Frequency", "@freqs Hz"), ("ASDR", "@asdr")],
            mode="vline",
        ),
        BoxZoomTool(dimensions="width"),
    )

    asdrs = get_asdr_vs_time(
        clean_data,
        raw_data,
        window_length=20,
        window_step=10,
        fftlength=fftlength,
        sample_rate=sample_rate,
        freq_low=freq_low,
        freq_high=freq_high,
        overlap=overlap,
    )
    asdrs_source = ColumnDataSource(
        {"time": [10 * (i + 1) for i in range(len(asdrs))], "asdr": asdrs}
    )
    p_asdrt = figure(
        height=300,
        width=800,
        title="Average ratio of clean strain to raw strain over time",
        x_axis_label="Time [s]",
        y_axis_label="Average ratio [Clean / Raw]",
        sizing_mode="scale_width",
        tools="reset",
    )
    r = p_asdrt.line(
        x="time",
        y="asdr",
        line_color=palette[3],
        line_width=2.3,
        line_alpha=0.8,
        source=asdrs_source,
    )
    p_asdrt.add_tools(
        HoverTool(
            renderers=[r],
            tooltips=[("Window center", "@time s"), ("ASDR", "@asdr")],
            mode="vline",
        ),
        BoxZoomTool(dimensions="width"),
    )

    return column(row(p_asd, p_asdr), p_asdrt), len(raw_fnames)


@typeo
def main(
    raw_data_dir: Path,
    clean_data_dir: Path,
    output_directory: Path,
    channels: ChannelList,
    sample_rate: float,
    fftlength: float,
    freq_low: Optional[float] = None,
    freq_high: Optional[float] = None,
    overlap: Optional[float] = None,
) -> None:
    """
    Build an HTML document analyzing a set of gravitational
    wave frame files cleaned using DeepClean. This includes
    plots of both the cleaned and uncleaned ASDs, as well as
    of the ratio of these ASDs plotted over the frequency
    range of interest. Included above these plots are the
    training and validation loss curves from DeepClean
    training as well as a box including any relevant logs
    or configs used to generate this run of DeepClean.

    Args:
        raw_data_dir:
            Directory containing the raw frame files containing
            the strain channel DeepClean was used to clean
        clean_data_dir:
            Directory containing the frame files produced by
            DeepClean with the cleaned strain channel (whose
            name should match the raw strain channel)
        output_directory:
            Directory to which HTML document should be written
            as `analysis.html`. Should also include any log files
            (ending `.log`) that are desired to be included in
            the plot.
        channels:
            A list of channel names used by DeepClean, with the
            strain channel first, or the path to a text file
            containing this list separated by newlines
        sample_rate:
            Rate at which the input data to DeepClean was sampled,
            in Hz
        fflength:
            Length of time, in seconds, over which to compute the
            ASDs of the cleaned and uncleaned data
        freq_low:
            The low end of the frequency range of interest for
            plotting the ASD ratio
        freq_high:
            The high end of the frequency range of interest
            for plotting the ASD ratio
        overlap:
            The amount of overlap, in seconds, between successive
            FFT windows for the ASD computation. If left as `None`,
            this will default to `fftlength / 2`.
    """

    # load the channels from a file if we specified one
    channels = get_channels(channels)

    asdr_plots, num_frames = analyze_test_data(
        raw_data_dir,
        clean_data_dir,
        output_directory,
        channels,
        sample_rate,
        fftlength,
        freq_low,
        freq_high,
        overlap,
    )

    header = Div(
        text=f"""
        <h1>DeepClean Sandbox Experiment Results</h1>
        <h2>Analysis on {num_frames} frames of test data</h2>
    """
    )

    train_curves = get_training_curves(output_directory)
    tabs = get_logs_box(output_directory)
    metadata = row(train_curves, tabs)
    layout = column(header, metadata, asdr_plots)
    save(
        layout,
        filename=output_directory / "analysis.html",
        title="DeepClean Results",
    )


if __name__ == "__main__":
    main()
