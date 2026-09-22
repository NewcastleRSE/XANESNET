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

"""Plotter for scalar value distributions."""

import logging
from itertools import repeat
from pathlib import Path

import numpy as np
from matplotlib.axes import Axes

from xanesnet.serialization.config import Config

from ..result import AnalysisResults
from ..utils import ScalarValue, iter_scalar_items, one_line_label
from .base import Plotter
from .common.formatting import format_decimal, shorten_label
from .common.layout import (
    finish_grid,
    method_grid,
    reserve_grid_headroom,
    save_figure,
    single_panel,
    style_grid_cell,
)
from .common.style import (
    COLOR_ACCENT_RED,
    FIGURE_SIZES,
    PlotSize,
    PlotStyle,
    add_legend,
    add_note,
    method_color,
    set_context_title,
)
from .registry import PlotterRegistry

_CLIP_IQR_SCALE: float = 1.5
_CLIP_MIN_SAMPLES: int = 20


@PlotterRegistry.register("scalar")
class ScalarPlotter(Plotter):
    """Create distribution plots and combined figures for scalar value keys.

    Per method this writes a histogram, box plot, and violin plot per key;
    four combined figures per key compare all methods. Histograms clip to a
    Tukey-fence range; box and violin plots always show the full range.

    Requires:
        Scalar collector output (optional).

    Args:
        plotter_type: Registered plotter name from the analysis configuration.
        latex_font: Render figures in a LaTeX-style serif font when ``True``.
        bins: Number of histogram bins.
        legend_position: Place the combined grouped-bar legend ``"inside"``
            the axes or ``"outside"`` it on the right.
        plot_size: Shared figure size profile: ``"small"`` or ``"default"``.
    """

    def __init__(
        self,
        plotter_type: str,
        bins: int,
        legend_position: str,
        latex_font: bool,
        plot_size: PlotSize,
    ) -> None:
        """Initialize a scalar distribution plotter."""
        super().__init__(plotter_type, latex_font=latex_font, plot_size=plot_size)
        self.bins = bins
        self.legend_position = legend_position

    def _plot(self, results: AnalysisResults, output_dir: Path) -> None:
        """Create per-method distribution plots and combined per-key figures.

        Args:
            results: Analysis pipeline outputs to plot.
            output_dir: Directory where the ``scalar_plots`` tree should be written.
        """
        root = output_dir / "scalar_plots"

        series: list[tuple[str, list[str], str, dict[str, list[ScalarValue]]]] = []
        key_order: list[str] = []

        for reader_idx, reader_selectors in enumerate(results.selectors):
            logging.info(f"    Predictions {reader_idx + 1}/{len(results.selectors)}.")

            for sel_idx, selector in enumerate(reader_selectors):
                logging.info(f"      Selector {sel_idx + 1}/{len(reader_selectors)}.")
                label = results.method_label(reader_idx, sel_idx)
                stream = results.collector_stream(reader_idx, sel_idx)

                values_by_key: dict[str, list[ScalarValue]] = {}
                for sample, record in zip(selector, stream if stream is not None else repeat({})):
                    scalars = dict(iter_scalar_items(sample))
                    scalars.update(iter_scalar_items(record))
                    for key, value in scalars.items():
                        values_by_key.setdefault(key, []).append(value)
                if not values_by_key:
                    continue

                color = method_color(len(series))
                series.append((label.dir_name, label.lines, color, values_by_key))
                for key in values_by_key:
                    if key not in key_order:
                        key_order.append(key)

        if not series:
            logging.info("    No scalar values collected, skipping.")
            return

        for dir_name, label_lines, color, values_by_key in series:
            combo_dir = root / dir_name
            subtitle = one_line_label(label_lines)

            for key, vals in values_by_key.items():
                key_dir = combo_dir / key
                key_dir.mkdir(parents=True, exist_ok=True)
                arr = np.array(vals)

                self._histogram(arr, key, subtitle, color, key_dir)
                self._boxplot(arr, key, subtitle, color, key_dir)
                self._violin(arr, key, subtitle, color, key_dir)

        combined_dir = root / "combined"
        combined_dir.mkdir(parents=True, exist_ok=True)
        for key in key_order:
            key_series: list[tuple[str, list[str], np.ndarray]] = []
            for _, label_lines, color, values_by_key in series:
                if key in values_by_key:
                    key_series.append((color, label_lines, np.array(values_by_key[key])))
            self._combined_grid(key, key_series, combined_dir)
            self._combined_grouped(key, key_series, combined_dir)
            self._combined_boxplot(key, key_series, combined_dir)
            self._combined_violin(key, key_series, combined_dir)

    def _histogram(self, arr: np.ndarray, key: str, subtitle: str, color: str, out: Path) -> None:
        """Write a histogram PDF for one scalar key.

        Bar heights report the percentage share of in-range samples per bin.

        Args:
            arr: One-dimensional scalar values with shape ``(N,)``.
            key: Scalar value key used for labels and output path context.
            subtitle: Plot subtitle text describing prediction and selector context.
            color: Method color used for the histogram bars.
            out: Directory where ``histogram.pdf`` should be written.
        """
        fig, ax = single_panel(self.style, "square")
        clip = _clip_limits(arr)
        lo, hi = _bin_range(arr, clip)
        heights, edges = _share_heights(arr, self.bins, lo, hi)
        width = edges[1] - edges[0]
        centers = (edges[:-1] + edges[1:]) / 2
        ax.bar(centers, heights, width=width, color=color, alpha=0.55)
        ax.set_xlim(lo, hi)
        ax.set_xlabel(key)
        ax.set_ylabel("Share (%)")
        set_context_title(ax, subtitle, self.style)
        _add_clip_note(ax, clip, self.style)
        _add_stats_text(ax, arr, self.style)
        save_figure(fig, out / "histogram.pdf")

    def _boxplot(self, arr: np.ndarray, key: str, subtitle: str, color: str, out: Path) -> None:
        """Write a box plot PDF for one scalar key.

        Args:
            arr: One-dimensional scalar values with shape ``(N,)``.
            key: Scalar value key used for labels and output path context.
            subtitle: Plot subtitle text describing prediction and selector context.
            color: Method color used for the box fill.
            out: Directory where ``boxplot.pdf`` should be written.
        """
        fig, ax = single_panel(self.style, "square")
        bp = ax.boxplot(arr, vert=True, patch_artist=True)
        bp["boxes"][0].set_facecolor(color)
        bp["boxes"][0].set_alpha(0.7)
        for lines in bp.values():
            for line in lines:
                line.set_linewidth(self.style.linewidth("box"))
        ax.set_ylabel(key)
        ax.set_xticklabels([""])
        set_context_title(ax, subtitle, self.style)
        save_figure(fig, out / "boxplot.pdf")

    def _violin(self, arr: np.ndarray, key: str, subtitle: str, color: str, out: Path) -> None:
        """Write a violin plot PDF for one scalar key.

        Args:
            arr: One-dimensional scalar values with shape ``(N,)``.
            key: Scalar value key used for labels and output path context.
            subtitle: Plot subtitle text describing prediction and selector context.
            color: Method color used for the violin bodies.
            out: Directory where ``violin.pdf`` should be written.
        """
        fig, ax = single_panel(self.style, "square")
        vp = ax.violinplot(arr, showmedians=True, showextrema=True)
        bodies = vp["bodies"]
        assert isinstance(bodies, list)
        for body in bodies:
            body.set_facecolor(color)
            body.set_alpha(0.7)
            body.set_linewidth(self.style.linewidth("box"))
        ax.set_ylabel(key)
        ax.set_xticks([1])
        ax.set_xticklabels([""])
        set_context_title(ax, subtitle, self.style)
        save_figure(fig, out / "violin.pdf")

    def _combined_grid(
        self,
        key: str,
        series: list[tuple[str, list[str], np.ndarray]],
        out: Path,
    ) -> None:
        """Write one combined figure with a per-method histogram grid.

        All cells share one x range, one y range, and one clip range so the
        cells stay directly comparable. Each cell annotates how many of its
        own values fall outside the shared display range.

        Args:
            key: Scalar value key used for labels and output path context.
            series: ``(color, label lines, values)`` per method in first-seen order.
            out: Directory where the combined grid PDF should be written.
        """
        n = len(series)

        all_values = np.concatenate([arr for _, _, arr in series])
        clip = _clip_limits(all_values)
        lo, hi = _bin_range(all_values, clip)

        fig, axes = method_grid(n, self.style, geometry="square")
        ncols = len(axes[0])

        for idx, (color, label_lines, arr) in enumerate(series):
            ax = axes[idx // ncols][idx % ncols]
            heights, edges = _share_heights(arr, self.bins, lo, hi)
            width = edges[1] - edges[0]
            centers = (edges[:-1] + edges[1:]) / 2
            ax.bar(centers, heights, width=width, color=color, alpha=0.8)
            ax.set_xlim(lo, hi)
            style_grid_cell(ax, label_lines, self.style)
            if clip is not None:
                clipped_i = int(np.count_nonzero((arr < lo) | (arr > hi)))
                _add_cell_clip_note(ax, (lo, hi, clipped_i), self.style)

        finish_grid(axes, n, key, "Share (%)", self.style)

        if clip is not None and clip[2] > 0:
            reserve_grid_headroom(axes, n)

        save_figure(fig, out / f"{key}_grid.pdf")

    def _combined_grouped(
        self,
        key: str,
        series: list[tuple[str, list[str], np.ndarray]],
        out: Path,
    ) -> None:
        """Write one combined figure with side-by-side per-method share bars.

        One bar per method sits next to the others inside each bin so the
        distributions stay comparable without stacking.

        Args:
            key: Scalar value key used for labels and output path context.
            series: ``(color, label lines, values)`` per method in first-seen order.
            out: Directory where the combined grouped PDF should be written.
        """
        all_values = np.concatenate([arr for _, _, arr in series])
        clip = _clip_limits(all_values)
        lo, hi = _bin_range(all_values, clip)
        bins = np.linspace(lo, hi, self.bins + 1)
        width = bins[1] - bins[0]
        centers = (bins[:-1] + bins[1:]) / 2

        n_methods = len(series)
        sub_width = width / n_methods
        offsets = (np.arange(n_methods) - (n_methods - 1) / 2) * sub_width

        fig, ax = single_panel(
            self.style,
            "square",
            width=max(FIGURE_SIZES["square"][0], 1.2 * n_methods),
            legend_position=self.legend_position,
        )
        for idx, (color, label_lines, arr) in enumerate(series):
            heights, _ = _share_heights(arr, self.bins, lo, hi)
            ax.bar(
                centers + offsets[idx],
                heights,
                width=sub_width,
                color=color,
                alpha=0.9,
                edgecolor="white",
                linewidth=self.style.linewidth("bar_edge"),
                label=one_line_label(label_lines),
            )
            ax.set_xlim(lo, hi)
        ax.set_xlabel(key)
        ax.set_ylabel("Share (%)")
        _add_clip_note(ax, clip, self.style)
        add_legend(ax, self.style, position=self.legend_position, ncol=2)
        save_figure(fig, out / f"{key}_grouped.pdf")

    def _combined_boxplot(
        self,
        key: str,
        series: list[tuple[str, list[str], np.ndarray]],
        out: Path,
    ) -> None:
        """Write one combined figure with side-by-side per-method box plots.

        All boxes share one axis so they stay directly comparable. The full
        value range is always displayed.

        Args:
            key: Scalar value key used for labels and output path context.
            series: ``(color, label lines, values)`` per method in first-seen order.
            out: Directory where the combined box plot PDF should be written.
        """
        n = len(series)

        fig, ax = single_panel(
            self.style,
            "square",
            width=max(FIGURE_SIZES["square"][0], 1.2 * n),
        )
        bp = ax.boxplot([arr for _, _, arr in series], positions=range(n), vert=True, patch_artist=True)
        for patch, (color, _, _) in zip(bp["boxes"], series):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        ax.set_xticks(range(n))
        _set_method_xticklabels(ax, [label_lines for _, label_lines, _ in series], self.style)
        ax.set_ylabel(key)
        save_figure(fig, out / f"{key}_boxplot.pdf")

    def _combined_violin(
        self,
        key: str,
        series: list[tuple[str, list[str], np.ndarray]],
        out: Path,
    ) -> None:
        """Write one combined figure with side-by-side per-method violin plots.

        All violins share one axis so they stay directly comparable. The full
        value range is always displayed.

        Args:
            key: Scalar value key used for labels and output path context.
            series: ``(color, label lines, values)`` per method in first-seen order.
            out: Directory where the combined violin PDF should be written.
        """
        n = len(series)

        fig, ax = single_panel(
            self.style,
            "square",
            width=max(FIGURE_SIZES["square"][0], 1.2 * n),
        )
        vp = ax.violinplot([arr for _, _, arr in series], positions=range(n), showmedians=True, showextrema=True)
        bodies = vp["bodies"]
        assert isinstance(bodies, list)
        for body, (color, _, _) in zip(bodies, series):
            body.set_facecolor(color)
            body.set_alpha(0.7)
        ax.set_xticks(range(n))
        _set_method_xticklabels(ax, [label_lines for _, label_lines, _ in series], self.style)
        ax.set_ylabel(key)
        save_figure(fig, out / f"{key}_violin.pdf")

    @property
    def signature(self) -> Config:
        """Return the scalar plotter signature.

        Returns:
            Configuration values needed to recreate this plotter.
        """
        signature = super().signature
        signature.update_with_dict({"bins": self.bins, "legend_position": self.legend_position})
        return signature


def _bin_range(arr: np.ndarray, clip: tuple[float, float, int] | None) -> tuple[float, float]:
    """Return lower and upper bin edges for one sample.

    Uses the clipped fence bounds when available and otherwise the data range,
    padded for degenerate data.

    Args:
        arr: One-dimensional scalar values with shape ``(N,)``.
        clip: Clip bounds from :func:`_clip_limits` or ``None``.

    Returns:
        ``(lo, hi)`` bin edges with ``lo < hi``.
    """
    lo = float(np.min(arr))
    hi = float(np.max(arr))
    if clip is not None:
        lo = max(lo, clip[0])
        hi = min(hi, clip[1])
    if not lo < hi:
        pad = 0.5 * max(abs(lo), 1.0)
        lo -= pad
        hi += pad
    return lo, hi


def _share_heights(arr: np.ndarray, n_bins: int, lo: float, hi: float) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-bin percentage shares for one sample.

    Args:
        arr: One-dimensional scalar values with shape ``(N,)``.
        n_bins: Number of histogram bins.
        lo: Lower bin edge.
        hi: Upper bin edge.

    Returns:
        ``(heights, edges)`` where ``heights`` holds the percentage share of
        in-range samples per bin and ``edges`` the ``n_bins + 1`` bin edges.
    """
    counts, edges = np.histogram(arr, bins=n_bins, range=(lo, hi))
    n_in = int(counts.sum())
    heights = counts / n_in * 100.0 if n_in > 0 else np.zeros_like(counts, dtype=float)
    return heights, edges


def _clip_limits(arr: np.ndarray) -> tuple[float, float, int] | None:
    """Compute robust display limits and the number of values outside them.

    Uses Tukey inner fences (``Q1 - 1.5 * IQR`` to ``Q3 + 1.5 * IQR``) so that
    outliers do not compress the interesting range; robust even when outliers
    make up a few percent of the sample. No values are removed; the clipped
    count is only used for on-figure annotations and summary statistics always
    use all values.

    Args:
        arr: One-dimensional scalar values with shape ``(N,)``.

    Returns:
        ``(lo, hi, clipped)`` fence bounds plus the number of values outside
        them, or ``None`` when clipping is not useful (too few samples or a
        degenerate interquartile range).
    """
    if len(arr) < _CLIP_MIN_SAMPLES:
        return None
    q1, q3 = np.quantile(arr, [0.25, 0.75])
    spread = _CLIP_IQR_SCALE * (q3 - q1)
    lo = float(q1 - spread)
    hi = float(q3 + spread)
    if not lo < hi:
        return None
    clipped = int(np.count_nonzero((arr < lo) | (arr > hi)))
    return lo, hi, clipped


def _clip_note(clip: tuple[float, float, int] | None) -> str:
    """Describe how many values fall outside the clipped display range.

    Args:
        clip: Clip bounds from :func:`_clip_limits` or ``None``.

    Returns:
        Annotation text, or an empty string when no clipping applies.
    """
    if clip is None or clip[2] == 0:
        return ""
    return f"{clip[2]} outliers"


def _add_clip_note(ax: Axes, clip: tuple[float, float, int] | None, style: PlotStyle) -> None:
    """Annotate an axis with the number of clipped values, when any.

    The note is anchored inside the plot at the upper-right corner.

    Args:
        ax: Matplotlib axis to annotate.
        clip: Clip bounds from :func:`_clip_limits` or ``None``.
        style: Rendering style for the figure.
    """
    note = _clip_note(clip)
    if not note:
        return
    add_note(ax, note, style, location="upper right", boxed=False, color=COLOR_ACCENT_RED)


def _add_cell_clip_note(ax: Axes, clip: tuple[float, float, int] | None, style: PlotStyle) -> None:
    """Annotate one grid cell with the number of clipped values, when any.

    The note sits inside the cell at the upper-right corner; callers reserve
    headroom above the bars so it never overlaps them.

    Args:
        ax: Matplotlib axis to annotate.
        clip: Clip bounds from :func:`_clip_limits` or ``None``.
        style: Rendering style for the figure.
    """
    note = _clip_note(clip)
    if not note:
        return
    add_note(ax, note, style, location="upper right", boxed=False, color=COLOR_ACCENT_RED)


def _set_method_xticklabels(ax: Axes, label_lines: list[list[str]], style: PlotStyle) -> None:
    """Label method tick positions with stacked label lines.

    Short label sets are horizontal; long sets are rotated so they do not
    overlap.

    Args:
        ax: Matplotlib axis to annotate.
        label_lines: Label lines per method in tick order.
        style: Rendering style for the figure.
    """
    labels = [shorten_label(lines, width=18) for lines in label_lines]
    if len(labels) <= 5:
        ax.set_xticklabels(labels, fontsize=style.fontsize("tick"))
    else:
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=style.fontsize("tick"))


def _add_stats_text(ax: Axes, arr: np.ndarray, style: PlotStyle) -> None:
    """Add sample count and summary statistics to a plot axis.

    Args:
        ax: Matplotlib axis to annotate.
        arr: One-dimensional scalar values with shape ``(N,)``.
        style: Rendering style for the figure.
    """
    text = (
        f"n={len(arr)}\n"
        f"mean={format_decimal(float(np.mean(arr)), 4)}\n"
        f"std={format_decimal(float(np.std(arr)), 4)}\n"
        f"median={format_decimal(float(np.median(arr)), 4)}"
    )
    add_note(ax, text, style, location="upper right")
