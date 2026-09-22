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

"""Rendering profiles and helpers shared by every plotter figure."""

from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import Any, Literal

from matplotlib import rc_context
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.legend import Legend
from matplotlib.text import Text

from .formatting import apply_decimal_tick_format
from .visual_constants import (
    _DEFAULT_MARKER_SIZE,
    CANONICAL_HEIGHT,
    CANVAS_PADDING,
    COLOR_ACCENT_GREEN,
    COLOR_ACCENT_RED,
    COLOR_BOX_EDGE,
    COLOR_PREDICTION,
    COLOR_TARGET,
    COLORBAR_RECTS,
    FIGURE_SIZES,
    FONT_SIZES,
    GRID_CELL_SIZES,
    GRID_GAPS,
    GRID_MARGINS,
    GRID_NOTE_HEADROOM,
    LATEX_FONT_PARAMS,
    LINE_WIDTHS,
    METHOD_COLORS,
    PANEL_RECT,
    SCATTER_SIZES,
    SPACING,
    TABLE_LABEL_WIDTH,
    PlotGeometry,
)

PlotSize = Literal["small", "default"]


@dataclass(frozen=True)
class PlotStyle:
    """Rendering scale applied consistently to one plotter's figures.

    Args:
        name: User-facing plot size name.
        figure_scale: Scale applied to base figure dimensions in inches.
        font_scale: Scale applied to shared font sizes in points.
        line_scale: Scale applied to shared line widths in points.
        marker_scale: Scale applied to shared marker sizes.
        layout_scale: Scale applied to layout padding and label spacing.
        savefig_dpi: Raster resolution used when a figure contains raster data.
        table_tight_layout_pad: Padding passed to Matplotlib table layouts.
    """

    name: PlotSize
    figure_scale: float
    font_scale: float
    line_scale: float
    marker_scale: float
    layout_scale: float
    savefig_dpi: int
    table_tight_layout_pad: float = 1.08

    def figsize(self, base: tuple[float, float]) -> tuple[float, float]:
        """Scale a free-form figure size while preserving its aspect ratio.

        Args:
            base: Base figure width and height in inches.

        Returns:
            Scaled figure width and height in inches.
        """
        return base[0] * self.figure_scale, base[1] * self.figure_scale

    def figure_size(
        self,
        geometry: PlotGeometry,
        *,
        width: float | None = None,
        height: float | None = None,
        width_margin: float = 0.0,
    ) -> tuple[float, float]:
        """Return a named canonical figure size.

        Args:
            geometry: Named geometry family from :data:`FIGURE_SIZES`.
            width: Optional base width override for content-driven composite
                figures. The canonical height is retained when omitted.
            height: Optional base height override for content-driven figures.
            width_margin: Additional base inches reserved to the right of the
                figure, for example for an outside legend.

        Returns:
            Scaled figure width and height in inches.
        """
        base_width, base_height = FIGURE_SIZES[geometry]
        return self.figsize(
            (
                (base_width if width is None else width) + width_margin,
                base_height if height is None else height,
            )
        )

    def legend_margin(self, position: str) -> float:
        """Return the base width reserved for an outside legend.

        Args:
            position: Configured legend position.

        Returns:
            Additional base inches needed when ``position`` is ``"outside"``.
        """
        return 1.6 if position == "outside" else 0.0

    def panel_rect(self) -> tuple[float, float, float, float]:
        """Return the fixed axes rectangle in figure coordinates.

        Returns:
            ``(left, bottom, right, top)`` panel bounds as figure fractions.
        """
        return PANEL_RECT

    def grid_margins(self) -> tuple[float, float, float, float]:
        """Return scaled grid margins as (left, right, bottom, top) inches.

        Returns:
            Scaled outer grid margins in inches.
        """
        scale = self.figure_scale
        return (
            GRID_MARGINS["left"] * scale,
            GRID_MARGINS["right"] * scale,
            GRID_MARGINS["bottom"] * scale,
            GRID_MARGINS["top"] * scale,
        )

    def grid_gaps(self) -> tuple[float, float]:
        """Return scaled horizontal and vertical grid gaps in inches.

        Returns:
            Scaled horizontal and vertical gaps in inches.
        """
        scale = self.figure_scale
        return GRID_GAPS["width"] * scale, GRID_GAPS["height"] * scale

    def canvas_padding(self) -> tuple[float, float, float, float]:
        """Return canvas padding as (left, right, bottom, top) in inches.

        Canvas padding is deliberately profile-independent: it is physical
        whitespace around the finished figure rather than part of the plotted
        geometry.

        Returns:
            Canvas padding in inches.
        """
        return (CANVAS_PADDING, CANVAS_PADDING, CANVAS_PADDING, CANVAS_PADDING)

    def grid_cell_size(self, geometry: Literal["energy", "square"]) -> tuple[float, float]:
        """Return the scaled size of one canonical grid cell.

        Args:
            geometry: Canonical geometry family for each cell.

        Returns:
            Cell width and height in inches.
        """
        return self.figsize(GRID_CELL_SIZES[geometry])

    def grid_figsize(
        self,
        geometry: Literal["energy", "square"],
        nrows: int,
        ncols: int,
        *,
        width_margin: float = 0.0,
        height_margin: float = 0.0,
    ) -> tuple[float, float]:
        """Return the page size for a grid of fixed-size canonical cells.

        The page includes shared outer margins and gaps in addition to the
        requested margins. It therefore grows as methods are added instead of
        shrinking cells to fit a page.

        Args:
            geometry: Canonical geometry family for each cell.
            nrows: Number of grid rows.
            ncols: Number of grid columns.
            width_margin: Extra base width in inches.
            height_margin: Extra base height in inches.

        Returns:
            Figure width and height in inches.
        """
        cell_width, cell_height = GRID_CELL_SIZES[geometry]
        gap_width, gap_height = GRID_GAPS["width"], GRID_GAPS["height"]
        margin_width = GRID_MARGINS["left"] + GRID_MARGINS["right"]
        margin_height = GRID_MARGINS["bottom"] + GRID_MARGINS["top"]
        return self.figsize(
            (
                cell_width * ncols + gap_width * max(ncols - 1, 0) + margin_width + width_margin,
                cell_height * nrows + gap_height * max(nrows - 1, 0) + margin_height + height_margin,
            )
        )

    def spacing(self, role: str) -> float:
        """Return a profile-scaled shared spacing value.

        Args:
            role: Key in :data:`SPACING`.

        Returns:
            Spacing value in the active profile's units.
        """
        return SPACING[role] * self.layout_scale

    def fontsize(self, role: str) -> float:
        """Return a scaled shared font size.

        Args:
            role: Key in :data:`FONT_SIZES`.

        Returns:
            Font size in points.
        """
        return FONT_SIZES[role] * self.font_scale

    def table_label_width(self) -> int:
        """Return the maximum rendered table-header label width.

        Returns:
            Maximum number of characters used for a table header.
        """
        return TABLE_LABEL_WIDTH

    def linewidth(self, role: str) -> float:
        """Return a scaled shared line width.

        Args:
            role: Key in :data:`LINE_WIDTHS`.

        Returns:
            Line width in points.
        """
        return LINE_WIDTHS[role] * self.line_scale

    def scatter_size(self, role: str) -> float:
        """Return a scaled scatter area.

        Args:
            role: Key in :data:`SCATTER_SIZES`.

        Returns:
            Marker area in points squared.
        """
        return SCATTER_SIZES[role] * self.marker_scale**2

    def context(self, latex_font: bool) -> AbstractContextManager[None]:
        """Return a Matplotlib context configured for this plot size.

        Args:
            latex_font: Whether to use the LaTeX-style serif font settings.

        Returns:
            Context manager restoring Matplotlib settings on exit.
        """
        params: dict[str, object] = {
            "font.size": self.fontsize("annotation"),
            "axes.labelsize": self.fontsize("axis_label"),
            "axes.titlesize": self.fontsize("title"),
            "axes.titlepad": self.spacing("title_pad"),
            "xtick.labelsize": self.fontsize("tick"),
            "ytick.labelsize": self.fontsize("tick"),
            "xtick.major.pad": self.spacing("tick_pad"),
            "ytick.major.pad": self.spacing("tick_pad"),
            "legend.fontsize": self.fontsize("legend"),
            "legend.borderpad": self.spacing("box_pad"),
            "legend.handlelength": self.spacing("legend_handlelength"),
            "legend.handletextpad": self.spacing("legend_handletextpad"),
            "legend.labelspacing": self.spacing("legend_labelspacing"),
            "legend.columnspacing": self.spacing("legend_columnspacing"),
            "legend.borderaxespad": self.spacing("legend_borderaxespad"),
            "axes.linewidth": self.linewidth("spine"),
            "lines.linewidth": self.linewidth("rc_line"),
            "lines.markersize": _DEFAULT_MARKER_SIZE * self.marker_scale,
            "patch.linewidth": self.linewidth("patch"),
            "savefig.dpi": self.savefig_dpi,
        }
        if latex_font:
            params.update(LATEX_FONT_PARAMS)
        return rc_context(params)


def _uniform_style(
    name: PlotSize,
    figure_scale: float,
    visual_scale: float,
    table_tight_layout_pad: float,
) -> PlotStyle:
    """Build a profile whose visual controls share one scale.

    Args:
        name: User-facing profile name.
        figure_scale: Scale applied to figure dimensions.
        visual_scale: Scale applied to fonts, lines, markers, and spacing.
        table_tight_layout_pad: Padding for Matplotlib table layouts.

    Returns:
        The configured plot style.
    """
    return PlotStyle(
        name,
        figure_scale,
        visual_scale,
        visual_scale,
        visual_scale,
        visual_scale,
        300,
        table_tight_layout_pad,
    )


PLOT_STYLES: dict[PlotSize, PlotStyle] = {
    # Default has the larger canvas while keeping typography deliberately compact.
    "default": _uniform_style("default", 1.00, 0.90, 0.80),
    # Small uses the smaller canvas with larger visual elements for readability.
    "small": _uniform_style("small", 0.70, 1.00, 0.90),
}


def get_plot_style(plot_size: PlotSize) -> PlotStyle:
    """Return the shared rendering style for a configured plot size.

    Args:
        plot_size: One of ``"small"`` or ``"default"``.

    Returns:
        Immutable rendering style for the requested plot size.
    """
    return PLOT_STYLES[plot_size]


def method_color(index: int) -> str:
    """Return the palette color for a method index (cycling).

    Args:
        index: Zero-based method index in first-seen order.

    Returns:
        Matplotlib-compatible color string.
    """
    return METHOD_COLORS[index % len(METHOD_COLORS)]


def style_axis(ax: Axes, style: PlotStyle) -> None:
    """Apply uniform axis-label spacing to one axes.

    Both axis labels and both tick sets use the same shared distances so the
    visual rhythm is consistent across plotter types.

    Args:
        ax: Matplotlib axis to style.
        style: Rendering style for the figure containing the axis.
    """
    label_pad = style.spacing("label_pad")
    ax.xaxis.labelpad = label_pad
    ax.yaxis.labelpad = label_pad
    ax.xaxis.label.set_fontsize(style.fontsize("axis_label"))
    ax.yaxis.label.set_fontsize(style.fontsize("axis_label"))
    ax.tick_params(labelsize=style.fontsize("tick"), pad=style.spacing("tick_pad"))
    apply_decimal_tick_format(ax)


def add_legend(
    ax: Axes,
    style: PlotStyle,
    *,
    position: str = "inside",
    loc: str | None = None,
    ncol: int = 1,
    **kwargs: Any,
) -> Legend:
    """Add a consistently styled legend to one axis.

    Args:
        ax: Axis whose labelled artists should be included.
        style: Rendering style for the figure containing the axis.
        position: Whether to place the legend inside the axes or outside it
            on the right.
        loc: Optional Matplotlib legend location. Defaults to ``"upper
            right"`` inside the axes and ``"upper left"`` outside it.
        ncol: Number of legend columns.
        **kwargs: Optional Matplotlib legend overrides.

    Returns:
        The created legend.
    """
    resolved_loc = loc or ("upper left" if position == "outside" else "upper right")
    legend_kwargs = _legend_defaults(style, resolved_loc, ncol)
    if position == "outside":
        legend_kwargs["bbox_to_anchor"] = (1.06, 1.0)
        setattr(ax.figure, "_xanesnet_external_legend", True)
    legend_kwargs.update(kwargs)
    legend = ax.legend(**legend_kwargs)
    if legend is None:
        raise RuntimeError("A legend could not be created because the axis has no labelled artists.")
    return legend


def add_figure_legend(
    fig: Figure,
    handles: list[Any],
    labels: list[str],
    style: PlotStyle,
    *,
    position: str = "inside",
    loc: str | None = None,
    ncol: int = 1,
    **kwargs: Any,
) -> Legend:
    """Add a consistently styled figure-level legend.

    Args:
        fig: Figure receiving the legend.
        handles: Legend artists.
        labels: Text labels corresponding to ``handles``.
        style: Rendering style controlling legend typography and spacing.
        position: Whether to place the legend inside the figure or outside it
            on the right.
        loc: Optional Matplotlib legend location. Defaults to ``"upper
            center"`` inside the figure and ``"upper left"`` outside it.
        ncol: Number of legend columns.
        **kwargs: Additional Matplotlib legend options.

    Returns:
        The created legend.
    """
    resolved_loc = loc or ("upper left" if position == "outside" else "upper center")
    legend_kwargs = _legend_defaults(style, resolved_loc, ncol)
    if position == "outside":
        legend_kwargs["bbox_to_anchor"] = (1.06, 1.0)
        setattr(fig, "_xanesnet_external_legend", True)
    legend_kwargs.update(kwargs)
    legend = fig.legend(handles, labels, **legend_kwargs)
    if legend is None:
        raise RuntimeError("A figure legend could not be created.")
    return legend


def _legend_defaults(style: PlotStyle, loc: str, ncol: int) -> dict[str, Any]:
    """Return the shared legend parameters.

    Args:
        style: Rendering style controlling legend typography and spacing.
        loc: Legend location passed to Matplotlib.
        ncol: Number of legend columns.

    Returns:
        Matplotlib legend keyword arguments.
    """
    return {
        "loc": loc,
        "ncol": ncol,
        "fontsize": style.fontsize("legend"),
        "framealpha": 0.9,
        "borderpad": style.spacing("box_pad"),
        "handlelength": style.spacing("legend_handlelength"),
        "handletextpad": style.spacing("legend_handletextpad"),
        "labelspacing": style.spacing("legend_labelspacing"),
        "columnspacing": style.spacing("legend_columnspacing"),
        "borderaxespad": style.spacing("legend_borderaxespad"),
    }


def add_note(
    ax: Axes,
    text: str,
    style: PlotStyle,
    *,
    location: Literal["upper left", "upper right", "lower left", "lower right"] = "upper right",
    alpha: float = 0.8,
    boxed: bool = True,
    color: str | None = None,
) -> Text:
    """Add a consistently styled note in one axis corner.

    Args:
        ax: Axis receiving the note.
        text: Note text.
        style: Rendering style controlling note typography and padding.
        location: Corner in which to place the note.
        alpha: Note background opacity when boxed.
        boxed: Whether to draw a background box.
        color: Optional note text color.

    Returns:
        The created Matplotlib text object.
    """
    anchors = {
        "upper left": (0.03, 0.97, "left", "top"),
        "upper right": (0.97, 0.97, "right", "top"),
        "lower left": (0.03, 0.03, "left", "bottom"),
        "lower right": (0.97, 0.03, "right", "bottom"),
    }
    x, y, horizontal_alignment, vertical_alignment = anchors[location]
    if boxed:
        return ax.text(
            x,
            y,
            text,
            transform=ax.transAxes,
            fontsize=style.fontsize("annotation"),
            horizontalalignment=horizontal_alignment,
            verticalalignment=vertical_alignment,
            color=color,
            bbox=dict(
                boxstyle=f"round,pad={style.spacing('box_pad')}",
                facecolor="white",
                alpha=alpha,
                edgecolor=COLOR_BOX_EDGE,
            ),
        )
    return ax.text(
        x,
        y,
        text,
        transform=ax.transAxes,
        fontsize=style.fontsize("annotation"),
        horizontalalignment=horizontal_alignment,
        verticalalignment=vertical_alignment,
        color=color,
    )


def set_context_title(
    ax: Axes,
    text: str,
    style: PlotStyle,
    *,
    location: Literal["left", "center", "right"] = "center",
) -> Text:
    """Set the shared context title above an axes.

    Args:
        ax: Axis receiving the title.
        text: Context text to display.
        style: Rendering style controlling title typography and padding.
        location: Horizontal title alignment.

    Returns:
        The created Matplotlib text object.
    """
    return ax.set_title(
        text,
        loc=location,
        fontsize=style.fontsize("context"),
        pad=style.spacing("title_pad"),
    )
