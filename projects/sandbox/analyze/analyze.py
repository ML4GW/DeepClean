import os
import re
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


def build_timeseries(fnames: str, channel: str, sample_rate: float):
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
    nperseg = int(fftlength * sample_rate)
    overlap = overlap or fftlength / 2
    noverlap = int(overlap * sample_rate)

    f, psd = welch(data, fs=sample_rate, nperseg=nperseg, noverlap=noverlap)
    return f, np.sqrt(psd)


def get_training_curves(output_directory: Path):
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


def get_logs_box(output_directory):
    with open(Path(__file__).parent / ".." / "pyproject.toml", "r") as f:
        text_box = PreText(
            text=f.read(),
            height=1,
            width=1,
            style={
                "overflow-y": "scroll",
                "height": "300px",
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
                        "height": "300px",
                        "overflow-x": "scroll",
                        "width": "600px",
                    },
                )
            panel = Panel(child=text_box, title=fname.split(".")[0].title())
            panels.append(panel)

    return Tabs(tabs=panels)


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
        ),
        BoxZoomTool(dimensions="width"),
    )
    p_asd.legend.click_policy = "hide"

    p_asdr = figure(
        height=300,
        width=600,
        title="Ratio of clean strain to raw strain",
        x_axis_label="Frequency [Hz]",
        y_axis_label="Ratio",
        tooltips=[("Frequency", "@freqs Hz"), ("ASDR", "@asdr")],
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
    p_asdr.add_tools(BoxZoomTool(dimensions="width"))

    return row(p_asd, p_asdr), len(raw_fnames)


@typeo
def main(
    raw_data_dir: Path,
    clean_data_dir: Path,
    output_directory: Path,
    channels: List[str],
    sample_rate: float,
    fftlength: float,
    freq_low: Optional[float] = None,
    freq_high: Optional[float] = None,
    overlap: Optional[float] = None,
) -> None:
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
    save(layout, filename=output_directory / "analysis.html")


if __name__ == "__main__":
    main()
