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

"""Figure layout helpers."""

import math
from pathlib import Path
from typing import Any, Literal, cast

from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure

from .formatting import apply_decimal_tick_format, shorten_label
from .style import (
    COLORBAR_RECTS,
    GRID_NOTE_HEADROOM,
    PlotGeometry,
    PlotStyle,
    style_axis,
)


def single_panel(
    style: PlotStyle,
    geometry: PlotGeometry,
    *,
    width: float | None = None,
    height: float | None = None,
    extra_bottom: float = 0.0,
    legend_position: str = "inside",
    **kwargs: Any,
) -> tuple[Figure, Axes]:
    """Create one axis using a named canonical figure geometry.

    The physical axes box is placed by :func:`finish_single_panel`, so labels,
    legends, and annotations cannot resize it. Extra Matplotlib keyword
    arguments are passed to :func:`matplotlib.pyplot.subplots`.

    Args:
        style: Rendering style controlling geometry and typography.
        geometry: Named canonical figure geometry.
        width: Optional base figure width in inches.
        height: Optional base figure height in inches.
        extra_bottom: Additional base inches reserved below the axes.
        legend_position: Configured legend position. An outside legend adds
            right-hand figure space while preserving the canonical axes box.
        **kwargs: Additional arguments passed to ``plt.subplots``.

    Returns:
        The new figure and its single axes.

    Raises:
        ValueError: If ``extra_bottom`` is negative.
    """
    figsize = style.figure_size(
        geometry,
        width=width,
        height=height,
        width_margin=style.legend_margin(legend_position),
    )
    if extra_bottom < 0:
        raise ValueError("extra_bottom must be non-negative")
    figsize = (figsize[0], figsize[1] + extra_bottom * style.figure_scale)
    fig, ax = plt.subplots(figsize=figsize, **kwargs)
    finish_single_panel(fig, ax, style, geometry, width=width, height=height, extra_bottom=extra_bottom)
    return fig, ax


def finish_single_panel(
    fig: Figure,
    ax: Axes,
    style: PlotStyle,
    geometry: PlotGeometry,
    *,
    box_aspect: float | tuple[float, float, float] | None = None,
    width: float | None = None,
    height: float | None = None,
    extra_bottom: float = 0.0,
) -> None:
    """Fix one panel's physical box and mark the figure as layout-complete.

    The rectangle is shared by every ordinary plot. Its dimensions therefore
    depend only on the named geometry and profile scale, never on plotted data
    or text extents. A three-component ``box_aspect`` is available for 3D
    axes, whose Matplotlib API does not accept the scalar aspect used by 2D
    axes.

    Args:
        fig: Figure containing the axes.
        ax: Axes whose physical box should be fixed.
        style: Rendering style controlling geometry and typography.
        geometry: Named canonical figure geometry.
        box_aspect: Optional Matplotlib box aspect. A tuple is used for 3D
            axes; a scalar is used for 2D axes.
        width: Optional base figure width in inches.
        height: Optional base figure height in inches.
        extra_bottom: Additional base inches reserved below the axes.
    """
    left, bottom, right, top = style.panel_rect()
    base_width, base_height = style.figure_size(geometry, width=width, height=height)
    extra_bottom_inches = extra_bottom * style.figure_scale
    figure_width, figure_height = fig.get_figwidth(), fig.get_figheight()
    rect = (
        left * base_width / figure_width,
        (extra_bottom_inches + bottom * base_height) / figure_height,
        (right - left) * base_width / figure_width,
        (top - bottom) * base_height / figure_height,
    )
    fig.subplots_adjust(
        left=rect[0],
        bottom=rect[1],
        right=rect[0] + rect[2],
        top=rect[1] + rect[3],
    )
    if box_aspect is None:
        ax.set_box_aspect((rect[3] * figure_height) / (rect[2] * figure_width))
    elif isinstance(box_aspect, tuple):
        cast(Any, ax).set_box_aspect(box_aspect)
    else:
        ax.set_box_aspect(box_aspect)
    style_axis(ax, style)
    setattr(fig, "_xanesnet_style", style)
    setattr(fig, "_xanesnet_fixed_layout", True)


def grid_shape(n_cells: int) -> tuple[int, int]:
    """Return the near-square row and column count for a number of cells.

    Args:
        n_cells: Number of cells that must fit into the grid.

    Returns:
        ``(nrows, ncols)`` with ``nrows * ncols >= n_cells``.
    """
    if n_cells < 1:
        raise ValueError("n_cells must be positive")
    ncols = math.ceil(math.sqrt(n_cells))
    nrows = math.ceil(n_cells / ncols)
    return nrows, ncols


def method_grid(
    n_cells: int,
    style: PlotStyle,
    geometry: Literal["energy", "square"] = "energy",
    *,
    width_margin: float = 0.0,
    height_margin: float = 0.0,
    sharex: bool = True,
    sharey: bool = True,
) -> tuple[Figure, list[list[Axes]]]:
    """Create a near-square grid of cells, one per method.

    Args:
        n_cells: Number of methods to show.
        style: Rendering style for the grid.
        geometry: Canonical geometry family for every cell. ``"energy"`` uses
            rectangular cells and ``"square"`` uses square cells.
        width_margin: Extra figure width in inches, for example for a colorbar.
        height_margin: Extra figure height in inches, for example for a legend.
        sharex: Whether all cells share one x range.
        sharey: Whether all cells share one y range.

    Returns:
        ``(figure, axes)`` where ``axes`` is always a nested row-major list of
        shape ``(nrows, ncols)``.
    """
    nrows, ncols = grid_shape(n_cells)
    fig, axes = _fixed_grid(
        nrows,
        ncols,
        style,
        geometry,
        width_margin=width_margin,
        height_margin=height_margin,
        sharex=sharex,
        sharey=sharey,
    )
    _set_grid_tick_visibility(axes, n_cells)
    return fig, axes


def matrix_grid(
    nrows: int,
    ncols: int,
    style: PlotStyle,
    geometry: Literal["energy", "square"] = "square",
    *,
    width_margin: float = 0.0,
    height_margin: float = 0.0,
    sharex: bool = True,
    sharey: bool = True,
) -> tuple[Figure, list[list[Axes]]]:
    """Create a fixed row-by-column grid with canonical cell dimensions.

    Args:
        nrows: Number of grid rows.
        ncols: Number of grid columns.
        style: Rendering style controlling geometry and typography.
        geometry: Canonical geometry family for each cell.
        width_margin: Extra figure width in inches.
        height_margin: Extra figure height in inches.
        sharex: Whether cells share the x-axis.
        sharey: Whether cells share the y-axis.

    Returns:
        The figure and nested row-major axes list.
    """
    fig, axes = _fixed_grid(
        nrows,
        ncols,
        style,
        geometry,
        width_margin=width_margin,
        height_margin=height_margin,
        sharex=sharex,
        sharey=sharey,
    )
    _set_grid_tick_visibility(axes, nrows * ncols)
    return fig, axes


def _fixed_grid(
    nrows: int,
    ncols: int,
    style: PlotStyle,
    geometry: Literal["energy", "square"],
    *,
    width_margin: float,
    height_margin: float,
    sharex: bool,
    sharey: bool,
) -> tuple[Figure, list[list[Axes]]]:
    """Build a grid from axes rectangles measured in physical inches.

    Args:
        nrows: Number of grid rows.
        ncols: Number of grid columns.
        style: Rendering style controlling geometry and typography.
        geometry: Canonical geometry family for each cell.
        width_margin: Extra figure width in inches.
        height_margin: Extra figure height in inches.
        sharex: Whether cells share the x-axis.
        sharey: Whether cells share the y-axis.

    Returns:
        The figure and nested row-major axes list.
    """
    fig = plt.figure(
        figsize=style.grid_figsize(
            geometry,
            nrows,
            ncols,
            width_margin=width_margin,
            height_margin=height_margin,
        )
    )
    fig_width, fig_height = fig.get_figwidth(), fig.get_figheight()
    cell_width, cell_height = style.grid_cell_size(geometry)
    gap_width, gap_height = style.grid_gaps()
    left, _, bottom, _ = style.grid_margins()

    axes: list[list[Axes]] = []
    first_ax: Axes | None = None
    for row_index in range(nrows):
        row: list[Axes] = []
        for col_index in range(ncols):
            x = (left + col_index * (cell_width + gap_width)) / fig_width
            y = (bottom + (nrows - 1 - row_index) * (cell_height + gap_height)) / fig_height
            rect = (x, y, cell_width / fig_width, cell_height / fig_height)
            ax = fig.add_axes(rect)
            if first_ax is not None:
                if sharex:
                    ax.sharex(first_ax)
                if sharey:
                    ax.sharey(first_ax)
            ax.set_box_aspect(cell_height / cell_width)
            if first_ax is None:
                first_ax = ax
            style_axis(ax, style)
            row.append(ax)
        axes.append(row)

    setattr(fig, "_xanesnet_fixed_layout", True)
    setattr(fig, "_xanesnet_style", style)
    return fig, axes


def finish_grid(
    axes: list[list[Axes]],
    n_cells: int,
    xlabel: str,
    ylabel: str,
    style: PlotStyle,
) -> None:
    """Hide unused grid cells and label the outer cells of a method grid.

    Tick labels are shown only on the first column and on the bottom-most used
    cell of each column. Every left-column cell receives the y label and every
    bottom-row cell the x label, which keeps labels next to their tick values.

    Args:
        axes: Nested axes list from :func:`method_grid`.
        n_cells: Number of cells that carry data, in row-major order.
        xlabel: Label for the shared x quantity.
        ylabel: Label for the shared y quantity.
        style: Rendering style for the grid.
    """
    nrows = len(axes)
    ncols = len(axes[0])
    for index in range(n_cells, nrows * ncols):
        axes[index // ncols][index % ncols].axis("off")

    _set_grid_tick_visibility(axes, n_cells)

    label_fontsize = style.fontsize("axis_label")
    label_pad = style.spacing("label_pad")

    for col in range(ncols):
        used_rows = [row for row in range(nrows) if row * ncols + col < n_cells]
        if used_rows:
            axes[used_rows[-1]][col].tick_params(labelbottom=True)

    for row in range(nrows):
        axes[row][0].set_ylabel(
            ylabel,
            fontsize=label_fontsize,
            labelpad=label_pad,
        )
    for col in range(ncols):
        used_rows = [row for row in range(nrows) if row * ncols + col < n_cells]
        if not used_rows:
            continue
        axes[used_rows[-1]][col].set_xlabel(
            xlabel,
            fontsize=label_fontsize,
            labelpad=label_pad,
        )


def style_grid_cell(ax: Axes, title_lines: list[str], style: PlotStyle) -> None:
    """Apply the shared grid-cell title and tick styling to one cell.

    The method label lines are joined into a single line so the prediction
    name and selector sit side by side instead of stacking and overlapping
    the cell above.

    Args:
        ax: Cell axis to style.
        title_lines: Method label lines used as the cell title.
        style: Rendering style for the grid.
    """
    title = shorten_label(title_lines)
    ax.set_title(
        title,
        fontsize=style.fontsize("title"),
        pad=style.spacing("title_pad"),
    )
    ax.tick_params(labelsize=style.fontsize("tick"), pad=style.spacing("tick_pad"))
    apply_decimal_tick_format(ax)


def add_grid_label(ax: Axes, label_lines: list[str], style: PlotStyle, *, color: str | None = None) -> None:
    """Place a short centered label in a grid cell without an axis frame.

    Args:
        ax: Grid cell to label.
        label_lines: Label parts to join and shorten.
        style: Rendering style controlling the label font.
        color: Optional label color.
    """
    ax.axis("off")
    ax.text(
        0.5,
        0.5,
        shorten_label(label_lines),
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=style.fontsize("title"),
        color=color,
    )


def _set_grid_tick_visibility(axes: list[list[Axes]], n_cells: int) -> None:
    """Show y ticks only in the first column and x ticks at each column base.

    Args:
        axes: Nested row-major grid axes.
        n_cells: Number of axes containing data.
    """
    nrows = len(axes)
    ncols = len(axes[0])
    for index in range(n_cells):
        row, col = divmod(index, ncols)
        ax = axes[row][col]
        bottom_row = max(
            (candidate for candidate in range(nrows) if candidate * ncols + col < n_cells),
            default=row,
        )
        ax.tick_params(labelleft=col == 0, labelbottom=row == bottom_row)


def reserve_grid_headroom(axes: list[list[Axes]], n_cells: int) -> None:
    """Reserve one shared band above data for compact cell notes.

    Args:
        axes: Nested row-major grid axes.
        n_cells: Number of axes containing data.
    """
    ncols = len(axes[0])
    used = [axes[index // ncols][index % ncols] for index in range(n_cells)]
    if not used:
        return
    limits = [ax.get_ylim() for ax in used]
    lower = min(limit[0] for limit in limits)
    upper = max(limit[1] for limit in limits)
    if not upper > lower:
        return
    upper += GRID_NOTE_HEADROOM * (upper - lower)
    for ax in used:
        ax.set_ylim(lower, upper)


def save_figure(
    fig: Figure,
    out: Path | PdfPages,
    style: PlotStyle | None = None,
    *,
    bbox_inches: str | None = None,
) -> None:
    """Save and close a figure as a file or a page in an open PDF.

    Args:
        fig: Figure to save.
        out: Destination path or open multi-page PDF writer.
        style: Optional style used for the shared canvas padding.
        bbox_inches: Optional Matplotlib bounding-box mode, used when an
            artist intentionally extends beyond the axes. Tight saves receive
            the same physical padding through ``pad_inches``.
    """
    resolved_style = style or getattr(fig, "_xanesnet_style", None)
    pad_figure(fig, resolved_style)
    if bbox_inches is None and getattr(fig, "_xanesnet_external_legend", False):
        bbox_inches = "tight"
    save_kwargs: dict[str, Any] = {}
    if bbox_inches == "tight" and resolved_style is not None:
        save_kwargs["pad_inches"] = resolved_style.canvas_padding()[0]
    if isinstance(out, PdfPages):
        out.savefig(fig, bbox_inches=bbox_inches, **save_kwargs)
    else:
        fig.savefig(out, bbox_inches=bbox_inches, **save_kwargs)
    plt.close(fig)


def pad_figure(fig: Figure, style: PlotStyle | None) -> None:
    """Add shared physical whitespace without changing axes-box dimensions.

    Args:
        fig: Figure whose canvas should be padded.
        style: Rendering style providing the padding, or ``None`` to skip it.
    """
    if style is None or getattr(fig, "_xanesnet_canvas_padded", False):
        return

    left, right, bottom, top = style.canvas_padding()
    old_width, old_height = fig.get_figwidth(), fig.get_figheight()
    positions = [ax.get_position().frozen() for ax in fig.axes]
    new_width = old_width + left + right
    new_height = old_height + bottom + top
    fig.set_size_inches(new_width, new_height, forward=True)
    for ax, position in zip(fig.axes, positions):
        ax.set_position(
            (
                (position.x0 * old_width + left) / new_width,
                (position.y0 * old_height + bottom) / new_height,
                position.width * old_width / new_width,
                position.height * old_height / new_height,
            )
        )
    setattr(fig, "_xanesnet_canvas_padded", True)


def add_colorbar(
    fig: Figure,
    mappable: Any,
    style: PlotStyle,
    *,
    orientation: Literal["horizontal", "vertical"] = "vertical",
    rect: tuple[float, float, float, float] | None = None,
    label: str = "count",
) -> Any:
    """Add a profile-styled colorbar in a caller-reserved rectangle.

    Args:
        fig: Figure receiving the colorbar axes.
        mappable: Matplotlib artist supplying the color scale.
        style: Rendering style controlling fonts and spacing.
        orientation: Colorbar orientation.
        rect: Optional figure-coordinate rectangle. Defaults to the shared
            horizontal rectangle; vertical colorbars require an explicit one.
        label: Colorbar label, or an empty string to omit it.

    Returns:
        The created Matplotlib colorbar.

    Raises:
        ValueError: If a vertical colorbar is requested without ``rect``.
    """
    if rect is None:
        try:
            rect = COLORBAR_RECTS[orientation]
        except KeyError as exc:
            raise ValueError("a rectangle is required for vertical colorbars") from exc
    colorbar_axis = fig.add_axes(rect)
    colorbar = fig.colorbar(mappable, cax=colorbar_axis, orientation=orientation)
    apply_decimal_tick_format(colorbar.ax)
    colorbar.ax.tick_params(labelsize=style.fontsize("tick"), pad=style.spacing("tick_pad"))
    if label:
        colorbar.set_label(label, fontsize=style.fontsize("axis_label"), labelpad=style.spacing("label_pad"))
    return colorbar
