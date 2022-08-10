import numpy as np
from bokeh.io import save as bokeh_save
from bokeh.plotting import figure
from bokeh.resources import Resources

from deepclean.viz import BACKGROUND_COLOR


class DeepCleanResources(Resources):
    @property
    def css_raw(self):
        return super().css_raw + [
            f""".bk-root {{
                    background-color: {BACKGROUND_COLOR};
                    border-color: {BACKGROUND_COLOR};
                }}
            """
        ]


def save(*args, **kwargs):
    kwargs["resources"] = DeepCleanResources(mode="cdn")
    bokeh_save(*args, **kwargs)


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
