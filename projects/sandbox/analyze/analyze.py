from pathlib import Path
from typing import Optional

import h5py
import numpy as np
from bokeh.layouts import layout
from bokeh.models import Div, Panel, PreText, Tabs
from gwpy.timeseries import TimeSeries

from deepclean.gwftools.channels import ChannelList, get_channels
from deepclean.gwftools.io import find
from deepclean.viz import ASD_UNITS, BACKGROUND_COLOR, WHITISH, plots
from deepclean.viz import utils as plot_utils
from hermes.typeo import typeo


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
                "width": "500px",
                "color": WHITISH,
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
                "width": "500px",
                "color": WHITISH,
            },
        )
        panel = Panel(child=text_box, title=fname.stem.title())
        panels.append(panel)

    return Tabs(tabs=panels)


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
    column_width: int = 590,
    min_border: int = 80,
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

    # not worth starting if we don't have any cleaned data
    # to analyze, so do a lazy check on this here
    clean_data_dir = clean_data_dir or output_directory / "cleaned"
    if not clean_data_dir.is_dir():
        raise ValueError(
            f"Cleaned data directory '{clean_data_dir}' does not exist"
        )

    # start by loading in the saved analyses from training
    with h5py.File(output_directory / "train_results.h5", "r") as f:
        losses = {"train_asdr": f["train_loss"][:]}
        grads = f["train_gradients"][:]
        train_coherences = f["train_coherences"][:]

        if "valid_loss" in f.keys():
            losses["valid_asdr"] = f["valid_loss"][:]
            valid_grads = f["valid_gradients"][:]
            grads = np.stack([grads, valid_grads])

    # plot the training and validation losses
    loss_plot = plot_utils.make_plot(
        title="Training Curves",
        x_axis_label="Epoch",
        y_axis_label="Loss",
        height=300 + min_border,
        width=column_width + min_border,
        tools="reset",
        min_border=min_border,
    )
    loss_plot = plots.plot_loss(loss_plot, hover_on="train_asdr", **losses)

    # next plot the coherence of each witness channel
    # with the strain channel, aggregated over batches
    # in the training set

    # TODO: 2 is the fftlength used for the coherences,
    # which we'll hardcode for the time being but shouldn't
    coherence_fftlength = 2
    step = 1 / coherence_fftlength
    freqs = np.arange(0, sample_rate / 2 + step, step)
    if freq_low is not None and freq_high is not None:
        mask = (freq_low <= freqs) & (freqs <= freq_high)
        freqs = freqs[mask]
        train_coherences = train_coherences[:, :, mask]
    elif freq_low is not None or freq_high is not None:
        raise ValueError(
            "freq_high and freq_low most either both "
            "be None or neither can be None"
        )

    # create 25-75 percentile confidence intervals
    # around the median for each channel
    coherences = {}
    for i, channel in enumerate(channels[1:]):
        coherence = train_coherences[:, i]
        bands = np.concatenate([coherence[1], coherence[3, ::-1]])
        coherences[channel] = [coherence[2], bands]

    title = "Channel wise coherence with strain and mean gradient magnitude"
    coherence_plot = plot_utils.make_plot(
        title=title,
        x_axis_label="Frequency [Hz]",
        height=700 + min_border,
        width=int(column_width * 1.5) + min_border,
        x_range=(freqs.min(), freqs.max()),
        y_range=channels[1:],
        min_border=min_border,
    )
    coherence_plot.yaxis.major_label_text_font_size = "6pt"
    coherence_plot.xgrid.grid_line_width = 0.8
    coherence_plot.xgrid.grid_line_alpha = 0.2
    coherence_plot = plots.plot_coherence(
        coherence_plot, frequencies=freqs, gradients=grads, **coherences
    )

    # now let's load in the cleaned and raw data and
    # plot both their individual ASDs as well as their
    # ASDR over the frequency range of interest
    fnames = sorted(clean_data_dir.iterdir())
    fnames = [f for f in fnames if f.name.startswith("STRAIN")]

    strain_channel = channels[0]
    clean_channel = strain_channel + "-CLEANED"
    clean_timeseries = TimeSeries.read(fnames, channel=clean_channel)
    clean_timeseries = clean_timeseries.resample(sample_rate)

    raw_data = find(
        [strain_channel], t0, duration, sample_rate, data_path=raw_data_path
    )
    raw_data = raw_data[channels[0]][: len(clean_timeseries)]
    raw_timeseries = TimeSeries(raw_data, t0=t0, dt=1 / sample_rate)

    duration = clean_timeseries.duration
    asd_plot = plot_utils.make_plot(
        title=f"ASD from {duration} data of {strain_channel}",
        height=300 + min_border,
        width=column_width + min_border,
        y_axis_label=f"ASD [{ASD_UNITS}]",
        x_axis_label="Frequency [Hz]",
        y_axis_type="log",
        tools="reset",
        min_border=min_border,
    )
    asd_plot = plots.plot_asd(
        asd_plot,
        fftlength,
        overlap,
        raw=raw_timeseries,
        deepcleaned=clean_timeseries,
    )

    asdr_plot = plot_utils.make_plot(
        title="ASD Ratio",
        height=300 + min_border,
        width=column_width + min_border,
        y_axis_label="ASD Ratio [Cleaned / Raw]",
        x_axis_label="Frequency [Hz]",
        tools="reset",
        min_border=min_border,
    )
    asdr_plot = plots.plot_asdr(
        asdr_plot,
        raw_timeseries,
        clean_timeseries,
        window_length=window_length,
        fftlength=fftlength,
        overlap=overlap,
        percentile=25,
        freq_low=freq_low,
        freq_high=freq_high,
    )

    # grab our configs and add a header
    header = Div(
        text=(
            f"<h1 style='color:{WHITISH}'>"
            "DeepClean Sandbox Experiment Results</h1>"
        )
    )
    tabs = get_logs_box(output_directory)

    # compile everything into a single page and save it
    grid = layout(
        [header],
        [loss_plot, tabs],
        [coherence_plot],
        [asd_plot, asdr_plot],
        background=BACKGROUND_COLOR,
    )
    plot_utils.save(
        grid,
        filename=output_directory / "analysis.html",
        title="DeepClean Results",
    )


if __name__ == "__main__":
    main()
