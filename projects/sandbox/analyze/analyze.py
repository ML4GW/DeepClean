import re
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional

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

from deepclean.gwftools.channels import ChannelList, get_channels
from deepclean.gwftools.io import find
from hermes.typeo import typeo

if TYPE_CHECKING:
    from gwpy.frequencyseries import FrequencySeries


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

    fnames = output_directory.iterdir()
    logs = filter(lambda f: f.suffix == ".log", fnames)
    for fname in logs:
        text_box = PreText(
            text=fname.read_text(),
            height=1,
            width=1,
            style={
                "overflow-y": "scroll",
                "height": "250px",
                "overflow-x": "scroll",
                "width": "600px",
            },
        )
        panel = Panel(child=text_box, title=fname.stem.title())
        panels.append(panel)

    return Tabs(tabs=panels)


def get_asdr_vs_time(
    raw_timeseries: TimeSeries,
    clean_timeseries: TimeSeries,
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
        slc = slice(i * step_size, (i + 1) * step_size)
        clean_asd = clean_timeseries[slc].asd(fftlength, overlap=overlap)
        raw_asd = raw_timeseries[slc].asd(fftlength, overlap=overlap)

        if freq_low is not None:
            freqs = clean_asd.frequencies.value
            mask = (freq_low <= freqs) & (freqs < freq_high)
            clean_asd = clean_asd[mask]
            raw_asd = raw_asd[mask]
        asdrs.append((clean_asd / raw_asd).mean())
    return asdrs


def plot_asds(raw_asd: "FrequencySeries", clean_asd: "FrequencySeries"):
    source = ColumnDataSource(
        {
            "raw": raw_asd.value,
            "clean": clean_asd,
            "freqs": raw_asd.frequencies,
        }
    )
    p = figure(
        height=300,
        width=600,
        title="ASD of Raw and Clean Strains",
        y_axis_type="log",
        x_axis_label="Frequency [Hz]",
        y_axis_label="ASD [Hz??????????]",
        sizing_mode="scale_width",
        tools="reset",
    )

    for i, asd in enumerate(["raw", "clean"]):
        r = p.line(
            x="freqs",
            y=asd,
            line_color=palette[i],
            line_width=2.3,
            line_alpha=0.8,
            legend_label=asd.title(),
            source=source,
        )

    # add a hovertool that always displays based on
    # the x-position of the mouse and a zoom tool that
    # only allows for horizontal zooming
    p.add_tools(
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

    # make the legend interactive so we can view
    # either of the asds on their own
    p.legend.click_policy = "hide"
    return p


def plot_asdr(
    raw_timeseries: TimeSeries,
    clean_timeseries: TimeSeries,
    window_length: float,
    fftlength: float,
    freq_low: Optional[float] = None,
    freq_high: Optional[float] = None,
):
    raw_spec = raw_timeseries.spectrogram(window_length, fftlength)
    clean_spec = clean_timeseries.spectrogram(window_length, fftlength)

    freqs = raw_spec.frequencies.value
    psdr = clean_spec.value / raw_spec.value
    asdr = psdr**0.5
    percentiles = [25, 50, 75]
    low, med, high = np.percentile(asdr, percentiles, axis=0)

    # TODO: use signal lib's frequency check, allow for multiple ranges
    if freq_low is not None and freq_high is not None:
        mask = (freq_low <= freqs) & (freqs <= freq_high)
        freqs = freqs[mask]
        low, med, high = low[mask], med[mask], high[mask]
    elif freq_low is None or freq_high is None:
        # TODO: make more explicit
        raise ValueError(
            "freq_low and freq_high must both be either None or float"
        )

    source = ColumnDataSource({"asdr": med, "freqs": freqs})
    p = figure(
        height=300,
        width=600,
        title="Ratio of clean strain to raw strain",
        x_axis_label="Frequency [Hz]",
        y_axis_label="Ratio [Clean / Raw]",
        sizing_mode="scale_width",
        tools="reset",
    )
    r = p.line(
        x="freqs",
        y="asdr",
        line_color=palette[2],
        line_width=2.3,
        line_alpha=0.8,
        source=source,
    )
    p.add_tools(
        HoverTool(
            renderers=[r],
            tooltips=[("Frequency", "@freqs Hz"), ("ASDR", "@asdr")],
            mode="vline",
        ),
        BoxZoomTool(dimensions="width"),
    )

    p.patch(
        x=np.concatenate([freqs, freqs[::-1]]),
        y=np.concatenate([high, low[::-1]]),
        line_color=palette[2],
        line_width=1.1,
        line_alpha=0.8,
        fill_color=palette[2],
        fill_alpha=0.3,
    )
    return p


def analyze_test_data(
    raw_timeseries: np.ndarray,
    clean_data_dir: Path,
    output_directory: Path,
    channels: List[str],
    sample_rate: float,
    window_length: float,
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

    # assume that the cleaned files represent a subset
    # of the raw files, so use those to get the filenames
    fnames = sorted(clean_data_dir.iterdir())

    ts = None
    num_frames = 0
    for f in fnames:
        if not f.name.startswith("STRAIN"):
            continue
        try:
            y = TimeSeries.read(f, channel=channels[0] + "-CLEANED")
        except ValueError:
            raise ValueError(
                "No channel {} in frame file {}".format(
                    channels[0], clean_data_dir / f.name
                )
            )
        y = y.resample(sample_rate)
        if ts is None:
            ts = y
        else:
            ts = ts.append(y)
        num_frames += 1

    clean_timeseries = ts
    clean_asd = clean_timeseries.asd(fftlength, overlap=overlap)

    raw_timeseries = raw_timeseries[: len(clean_timeseries)]
    raw_timeseries = TimeSeries(raw_timeseries, dt=1 / sample_rate)
    raw_asd = raw_timeseries.asd(fftlength, overlap=overlap)

    p_asd = plot_asds(raw_asd, clean_asd)
    p_asdr = plot_asdr(
        raw_timeseries,
        clean_timeseries,
        window_length,
        fftlength,
        freq_low,
        freq_high,
    )
    return row(p_asd, p_asdr), num_frames


@typeo
def main(
    raw_data_path: Path,
    output_directory: Path,
    channels: ChannelList,
    t0: int,
    duration: int,
    sample_rate: float,
    window_length: float,
    fftlength: float,
    clean_data_dir: Optional[Path] = None,
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
    clean_data_dir = clean_data_dir or output_directory / "cleaned"

    raw_data = find(
        channels[:1], t0, duration, sample_rate, data_path=raw_data_path
    )
    raw_data = raw_data[channels[0]]

    asdr_plots, num_frames = analyze_test_data(
        raw_data,
        clean_data_dir,
        output_directory,
        channels=channels,
        sample_rate=sample_rate,
        window_length=window_length,
        fftlength=fftlength,
        freq_low=freq_low,
        freq_high=freq_high,
        overlap=overlap,
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
