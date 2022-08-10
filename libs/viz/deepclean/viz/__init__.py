from bokeh.io import curdoc
from bokeh.palettes import Set1_8 as palette
from bokeh.palettes import Turbo256 as spectrum
from bokeh.themes import Theme

palette = palette[1:3] + (palette[0],) + palette[3:]
WHITISH = "#aaaaaa"
BACKGROUND_COLOR = "#222b38"

SUPERSCRIPT_1 = "\xB9"
SUPERSCRIPT_2 = "\u00B2"
SUPERSCRIPT_NEGATIVE = "\u207B"
SUPERSCRIPT_SLASH = "\u141F"
PLUS_MINUS = "\u207A/\u208B"
MU = "\u03BC"
SIGMA_LOWER = "\u03C3"
SIGMA_UPPER = "\u03A3"
GRADIENT = "\u2207"

PSD_UNITS = "Hz{}{}".format(SUPERSCRIPT_NEGATIVE, SUPERSCRIPT_1)
ASD_UNITS = PSD_UNITS + SUPERSCRIPT_SLASH + SUPERSCRIPT_2

theme = {
    "attrs": {
        "Figure": {
            "background_fill_color": "#292630",
            "background_fill_alpha": 0.5,
            "border_fill_color": BACKGROUND_COLOR,
            "outline_line_color": WHITISH,
            "outline_line_width": 0.1,
            "height": 400,
            "width": 700,
        },
        "Grid": {"grid_line_color": WHITISH, "grid_line_width": 1.3},
        "Axis": {
            "major_label_text_color": WHITISH,
            "major_label_text_font_size": "12pt",
            "axis_label_text_color": WHITISH,
            "axis_label_text_font_size": "14pt",
        },
        "Title": {"text_color": WHITISH, "text_font_size": "16pt"},
        "Legend": {
            "background_fill_color": BACKGROUND_COLOR,
            "label_text_color": WHITISH,
        },
    },
    "line_defaults": {
        "line_width": 2.3,
        "line_alpha": 1.0,
        "line_color": palette[0],
    },
}

curdoc().theme = Theme(json=theme)
