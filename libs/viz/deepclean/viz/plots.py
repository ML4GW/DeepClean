from typing import List, Optional, Union

import numpy as np
from bokeh.models import BoxZoomTool, ColumnDataSource, HoverTool
from bokeh.plotting import Figure

from deepclean.viz import palette, spectrum


def plot_loss(
    p: Figure, hover_on: Optional[str] = None, **metrics: np.ndarray
) -> Figure:
    if hover_on is not None and hover_on not in metrics:
        raise ValueError(f"No metric {hover_on} to use with hover tool")
    elif hover_on is None:
        hover_on = list(metrics)[0]

    x_metric = list([i for i in metrics if i.startswith("x_")])
    if len(x_metric) == 0:
        x = np.arange(len(metrics[hover_on]))
        x_metric = p.xaxis.axis_label or "epoch"
    else:
        x = metrics.pop(x_metric)

    data = {"x": x}
    data.update(metrics)
    source = ColumnDataSource(data)
    colors = {metric: color for metric, color in zip(metrics, palette)}

    tooltips = [(x_metric.title(), "@x")]
    for metric in metrics:
        # TODO: add secondary y axis functionality
        # if metric.startswiths("secondary_"):
        #     if len(p.extra_y_ranges) == 0:
        metric_title = metric.replace("_", " ").title()
        p.line(
            "x",
            metric,
            line_color=colors[metric],
            legend_label=metric_title,
            source=source,
            name=metric,
        )
        tooltips.append((metric_title, f"@{metric}"))

    hover = HoverTool(
        mode="vline",
        line_policy="nearest",
        renderers=p.select(hover_on),
        tooltips=tooltips,
    )
    zoom = BoxZoomTool(dimensions="width")
    p.add_tools(hover, zoom)
    return p


def _make_ridge(channel, x, scale):
    return list(zip([channel] * len(x), x * scale))


def plot_coherence(
    p: Figure,
    frequencies: np.ndarray,
    scale: float = 1.5,
    **channels: Union[np.ndarray, List[np.ndarray]],
) -> Figure:
    lines, patches = [], []
    for channel, arrs in channels.items():
        if isinstance(arrs, np.ndarray):
            lines.appennd(arrs)
        else:
            lines.append(arrs[0])
            if len(arrs) > 1:
                patches.append(arrs[1:])

    # sort the channels by their integrated coherence
    # over all frequencies
    idx = np.argsort([i.sum() for i in lines])
    channels = list(channels)
    channels = [channels[i] for i in idx]
    lines = [lines[i] for i in idx]
    if len(patches) > 0:
        patches = [patches[i] for i in idx]

    # give each channel a unique color from our broader
    # color spectrum, spaced out as much as possible
    spectrum_stride = len(spectrum) // len(channels)
    colors = spectrum[::spectrum_stride]

    # add all the central lines to a data source,
    # formatted so they'll work with the categorical y range
    line_data = {"x": frequencies}
    for chan, arr in zip(channels, lines):
        if len(arr) != len(frequencies):
            raise ValueError(
                "Length of coherence {} for channel {} doesn't "
                "match length of frequency array {}".format(
                    len(arr), chan, len(frequencies)
                )
            )
        line_data[chan] = _make_ridge(chan, arr, scale)

        # add an additional entry to the data source to keep
        # track of the raw values for hovering purposes
        line_data[chan + "_hover"] = arr
    line_source = ColumnDataSource(line_data)

    # we'll plot all our patch data on a single glyph
    # since we don't care about hovering for them
    patch_x = np.concatenate([frequencies, frequencies[::-1]])
    patch_data = {"xs": [], "ys": [], "color": [], "alpha": []}
    for chan, patch, color in zip(channels, patches, colors):
        for i, y in enumerate(patch):
            if len(y) != (len(frequencies) * 2):
                raise ValueError(
                    "Length of coherence patch {} for channel {} doesn't "
                    "match length of frequency array {}".format(
                        len(y), chan, len(frequencies)
                    )
                )

            y = _make_ridge(chan, y, scale)
            patch_data["xs"].append(patch_x)
            patch_data["ys"].append(y)
            patch_data["color"].append(color)
            patch_data["alpha"].append(0.2 * 0.5**i)
    patch_source = ColumnDataSource(patch_data)

    # make some adjustments to the y-axis to accommodate
    # the sorted categorical values
    p.y_range.factors = channels
    p.y_range.range_padding = 0.1
    p.yaxis.major_label_orientation = np.pi / 8

    # plot each line individually and give it its
    # own hovertool for inspection
    for channel, color in zip(channels, colors):
        r = p.line(
            "x", channel, line_color=color, line_width=1.5, source=line_source
        )
        hover = HoverTool(
            renderers=[r],
            line_policy="nearest",
            tooltips=[
                ("Frequency", "@x"),
                ("Coherence", f"@{{{channel}_hover}}"),
            ],
        )
        p.add_tools(hover)

    # plot our patches if we have them
    if len(patches) > 0:
        p.patches(
            "xs",
            "ys",
            line_color="color",
            fill_color="color",
            line_width=0.8,
            fill_alpha="alpha",
            line_alpha=0.4,
            source=patch_source,
        )
    return p


def plot_asd(
    p: Figure,
    fftlength: float,
    overlap: Optional[float] = None,
    hover_on: Optional[str] = None,
    **timeseries,
) -> Figure:
    if hover_on is not None and hover_on not in timeseries:
        raise ValueError(f"No ASD '{hover_on}' to use with hover tool")
    elif hover_on is None:
        hover_on = list(timeseries)[0]

    ts = timeseries.pop(hover_on)
    asd = ts.asd(fftlength, overlap=overlap, method="median")

    # use the `hover_on` timeseries' asd as the
    # frequency reference for the remaining timeseries
    df = asd.df
    freqs = asd.frequencies

    data = {hover_on: asd.value, "x": freqs}
    colors = {hover_on: palette[0]}
    for color, (name, ts) in zip(palette[1:], timeseries.items()):
        asd = ts.asd(fftlength, overlap=overlap, method="median")
        if asd.df != df:
            asd = asd.resample(df)

        data[name] = asd
        colors[name] = color
    source = ColumnDataSource(data)

    tooltips = [("Frequency", "@x Hz")]
    for name in [hover_on] + list(timeseries):
        title = name.replace("_", " ").title()
        p.line(
            "x",
            name,
            line_color=colors[name],
            legend_label=title,
            source=source,
            name=name,
        )
        tooltips.append((title, f"@{name}"))

    hover = HoverTool(
        mode="vline",
        line_policy="nearest",
        renderers=p.select(hover_on),
        tooltips=tooltips,
    )
    zoom = BoxZoomTool(dimensions="width")
    p.add_tools(hover, zoom)
    return p


def plot_asdr(
    p: Figure,
    raw_timeseries,
    clean_timeseries,
    window_length: float,
    fftlength: float,
    overlap: Optional[float] = None,
    percentile: Union[float, List[float]] = 25,
    freq_low: Optional[float] = None,
    freq_high: Optional[float] = None,
) -> Figure:
    raw_spec = raw_timeseries.spectrogram(
        window_length, fftlength, overlap=overlap
    )
    clean_spec = clean_timeseries.spectrogram(
        window_length, fftlength, overlap=overlap
    )

    freqs = raw_spec.frequencies.value
    asdr = (clean_spec.value / raw_spec.value) ** 0.5

    # TODO: use signal lib's frequency check, allow for multiple ranges
    if freq_low is not None and freq_high is not None:
        mask = (freq_low <= freqs) & (freqs <= freq_high)
        freqs = freqs[mask]
        asdr = asdr[:, mask]
    elif freq_low is None or freq_high is None:
        # TODO: make more explicit
        raise ValueError(
            "freq_low and freq_high must both be either None or float"
        )

    median_asdr = np.median(asdr, axis=0)
    line_source = ColumnDataSource({"x": freqs, "y": median_asdr})

    try:
        iter(percentile)
    except Exception:
        percentile = [percentile]

    patch_data = {"xs": [], "ys": [], "legend": [], "alpha": []}
    patch_x = np.concatenate([freqs, freqs[::-1]])
    last_low = last_high = None
    for i, q in enumerate(sorted(percentile, reverse=True)):
        low = np.percentile(asdr, q, axis=0)
        high = np.percentile(asdr, 100 - q, axis=0)
        if last_low is None:
            y = np.concatenate([low, high[::-1]])
            x = patch_x
        else:
            y = np.concatenate(
                [low, last_low[::-1], [None], last_high, last_high[::-1]]
            )
            x = np.concatenate([patch_x, [None], patch_x])

        band = int(100 - 2 * q)
        patch_data["xs"].append(x)
        patch_data["ys"].append(y)
        patch_data["legend"].append(f"{band}% confidence band")
        patch_data["alpha"].append(0.2 * 0.5**i)
    patch_source = ColumnDataSource(patch_data)

    r = p.line("x", "y", source=line_source)
    p.patches(
        "xs",
        "ys",
        fill_color=palette[0],
        fill_alpha="alpha",
        line_width=1.0,
        line_alpha=0.6,
        legend_field="legend",
        source=patch_source,
    )

    hover = HoverTool(
        renderers=[r],
        tooltips=[("Frequency", "@x"), ("ASDR", "@y")],
        mode="vline",
    )
    zoom = BoxZoomTool(dimensions="width")
    p.add_tools(hover, zoom)

    return p
