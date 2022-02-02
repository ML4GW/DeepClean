import os
from pathlib import Path
from typing import List, Optional

import numpy as np
from bokeh.io import save
from bokeh.layouts import row
from bokeh.models import ColumnDataSource, HoverTool
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

    psd = welch(data, fs=sample_rate, nperseg=nperseg, noverlap=noverlap)
    return np.sqrt(psd)


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
    fnames = sorted(os.listdir(clean_data_dir))
    raw_fnames = [raw_data_dir / f for f in fnames]
    clean_fnames = [clean_data_dir / f for f in fnames]

    raw_data = build_timeseries(raw_fnames, channels[0], sample_rate)
    clean_data = build_timeseries(clean_fnames, channels[0], sample_rate)

    raw_asd = make_asd(raw_data, sample_rate, fftlength, overlap)
    clean_asd = make_asd(clean_data, sample_rate, fftlength, overlap)

    nperseg = int(fftlength * sample_rate)
    freqs = np.linspace(0, sample_rate / 2, nperseg // 2 + 1)
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
        height=600,
        width=600,
        title="ASD of Raw and Clean Strains",
        y_axis_type="log",
        x_axis_label="Frequency [Hz]",
        y_axis_label="ASD [Hz⁻¹]",
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
                ("Frequency", "@freq Hz"),
                ("Power of raw strain", "@raw"),
                ("Power of clean strain", "@clean"),
            ],
        )
    )

    p_asdr = figure(
        height=600,
        width=600,
        title="Ratio of clean strain to raw strain",
        x_axis_label="Frequency [Hz]",
        y_axis_label="Ratio",
        tooltips=[("Frequency", "@freq Hz"), ("ASDR", "asdr")],
    )
    p_asdr.line(
        x="freqs",
        y="asdr",
        line_color=palette[2],
        line_width=2.3,
        line_alpha=0.8,
        source=asdr_source,
    )

    layout = row(p_asd, p_asdr)
    save(layout, filename=output_directory / "analysis.html")


if __name__ == "__main__":
    main()
