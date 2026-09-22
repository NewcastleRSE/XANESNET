# SPDX-License-Identifier: GPL-3.0-or-later
#
# XANESNET
#
# Authors:  Hendrik Junkawitsch, Tom J. Penfold, Tom W. Pope, C. D. Rankine, B. Li
#
# This program is free software: you can redistribute it and/or modify it under the terms of the
# GNU General Public License as published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without
# even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with this program.
# If not, see <https://www.gnu.org/licenses/>.
#
# Citations:
#   ...

"""Shared visual constants used by the analysis plotters."""

from typing import Literal

METHOD_COLORS: list[str] = [
    "#0072b2",
    "#e69f00",
    "#009e73",
    "#d55e00",
    "#cc79a7",
    "#56b4e9",
    "#f0e442",
    "#7a5195",
    "#8c564b",
    "#17becf",
    "#7f7f7f",
    "#d62728",
    "#bcbd22",
    "#e7298a",
    "#000075",
    "#9a6324",
    "#008080",
    "#ff9896",
    "#c5b0d5",
    "#98df8a",
]
COLOR_TARGET: str = "#5a5a5a"
COLOR_PREDICTION: str = "#e69f00"
COLOR_ACCENT_RED: str = "#e85651"
COLOR_ACCENT_GREEN: str = "#3f9e6e"
COLOR_BOX_EDGE: str = "#bbbbbb"

CANONICAL_HEIGHT: float = 4.0

PlotGeometry = Literal["energy", "square", "spectra", "structure_page"]

FIGURE_SIZES: dict[PlotGeometry, tuple[float, float]] = {
    "energy": (6.4, CANONICAL_HEIGHT),
    "square": (CANONICAL_HEIGHT, CANONICAL_HEIGHT),
    "spectra": (10.0, CANONICAL_HEIGHT),
    "structure_page": (12.0, CANONICAL_HEIGHT),
}

PANEL_RECT: tuple[float, float, float, float] = (0.18, 0.18, 0.94, 0.94)

GRID_MARGINS: dict[str, float] = {
    "left": 0.55,
    "right": 0.35,
    "bottom": 0.55,
    "top": 0.45,
}
GRID_GAPS: dict[str, float] = {
    "width": 0.18,
    "height": 0.42,
}
GRID_NOTE_HEADROOM: float = 0.20
TABLE_LABEL_WIDTH: int = 24
CANVAS_PADDING: float = 1.0

COLORBAR_RECTS: dict[str, tuple[float, float, float, float]] = {
    "horizontal": (0.24, 0.08, 0.64, 0.025),
}

GRID_CELL_SIZES: dict[Literal["energy", "square"], tuple[float, float]] = {
    "energy": (3.2, 2.4),
    "square": (2.4, 2.4),
}

SPACING: dict[str, float] = {
    "label_pad": 5.0,
    "tick_pad": 3.0,
    "title_pad": 4.0,
    "box_pad": 0.35,
    "table_pad": 0.12,
    "legend_handlelength": 1.6,
    "legend_handletextpad": 0.4,
    "legend_borderaxespad": 0.35,
    "legend_labelspacing": 0.3,
    "legend_columnspacing": 1.0,
}

FONT_SIZES: dict[str, float] = {
    "axis_label": 10.0,
    "title": 10.0,
    "tick": 8.0,
    "legend": 10.0,
    "context": 10.0,
    "annotation": 8.0,
    "table": 10.0,
}

LINE_WIDTHS: dict[str, float] = {
    "main": 2.0,
    "secondary": 1.6,
    "light": 1.4,
    "emphasis": 1.8,
    "reference": 1.0,
    "identity": 1.2,
    "grid_reference": 0.8,
    "legend": 1.4,
    "bar_edge": 0.4,
    "box": 1.0,
    "atom": 0.5,
    "spine": 0.8,
    "target_ring": 2.0,
    "scale_bar": 1.2,
    "patch": 1.0,
    "rc_line": 1.5,
}

_DEFAULT_MARKER_SIZE: float = 6.0

SCATTER_SIZES: dict[str, float] = {
    "error_correlation": 6.0,
    "pca": 18.0,
    "structure_atom": 140.0,
    "structure_target": 420.0,
}

LATEX_FONT_PARAMS: dict[str, object] = {
    "font.family": "serif",
    "font.serif": ["STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
}


__all__ = [
    "CANONICAL_HEIGHT",
    "CANVAS_PADDING",
    "COLOR_ACCENT_GREEN",
    "COLOR_ACCENT_RED",
    "COLOR_BOX_EDGE",
    "COLOR_PREDICTION",
    "COLOR_TARGET",
    "COLORBAR_RECTS",
    "FIGURE_SIZES",
    "FONT_SIZES",
    "GRID_CELL_SIZES",
    "GRID_GAPS",
    "GRID_MARGINS",
    "GRID_NOTE_HEADROOM",
    "LATEX_FONT_PARAMS",
    "LINE_WIDTHS",
    "METHOD_COLORS",
    "PANEL_RECT",
    "PlotGeometry",
    "SCATTER_SIZES",
    "SPACING",
    "TABLE_LABEL_WIDTH",
]
