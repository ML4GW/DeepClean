import numpy as np
from bokeh.plotting import figure


def make_plot(tools="", **kwargs):
    p = figure(tools=tools, **kwargs)
    p.toolbar.autohide = True
    return p


def make_patch(hi: np.ndarray, lo: np.ndarray, x: np.ndarray):
    x = np.concatenate([x, x[::-1]])
    y = np.concatenate([hi, lo[::-1]])
    return x, y


# def summarize_with_errors(
#     x: np.ndarray,
#     average: str ="mean",
#     axis: int = -1,
#     errors: Optional[List[float]] = None
# ):
#     if average == "mean":
#         reduced = x.mean(axis=axis)
#         if errors is not None:
#             std = x.std(axis=axis)
#     elif average == "median":
#         reduced =
