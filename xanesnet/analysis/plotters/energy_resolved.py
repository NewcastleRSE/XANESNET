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

"""Plotter for energy-resolved loss curves."""

import logging
from pathlib import Path
from typing import Any

import numpy as np
from matplotlib.axes import Axes

from xanesnet.serialization.config import Config

from ..result import AnalysisResults
from ..utils import one_line_label
from .base import Plotter
from .common.formatting import apply_decimal_tick_format
from .common.layout import (
    finish_grid,
    method_grid,
    save_figure,
    single_panel,
    style_grid_cell,
)
from .common.style import PlotSize, add_legend, method_color, set_context_title
from .registry import PlotterRegistry

Curve = tuple[np.ndarray, np.ndarray]


@PlotterRegistry.register("energy_resolved_loss")
class EnergyResolvedLossPlotter(Plotter):
    """Plot average energy-resolved loss curves per method and combined.

    Writes one figure per (method, loss key) showing the mean curve with its
    standard-deviation band, plus a per-method grid and an overlay of all mean
    curves.

    Requires:
        Per-channel loss statistics: provided by the ``vector`` aggregator.

    Args:
        plotter_type: Registered plotter name from the analysis configuration.
        latex_font: Render figures in a LaTeX-style serif font when ``True``.
        y_scale: Y-axis scaling, either ``"linear"`` or ``"log"``.
        start_index: First channel index to plot; ``None`` starts at the first channel.
        end_index: One-past-the-end channel index to plot; ``None`` plots to the last channel.
        y_zoom: Optional upper y-axis limit for zooming in around zero. On a
            logarithmic axis, a positive lower limit is computed instead of zero.
        keys: Energy-resolved loss keys to plot. ``None`` plots every collected
            key.
        legend_position: Place curve legends ``"inside"`` the axes or
            ``"outside"`` them on the right.
        plot_size: Shared figure size profile: ``"small"`` or ``"default"``.
    """

    def __init__(
        self,
        plotter_type: str,
        y_scale: str,
        start_index: int | None,
        end_index: int | None,
        y_zoom: float | None,
        keys: list[str] | None,
        legend_position: str,
        latex_font: bool,
        plot_size: PlotSize,
    ) -> None:
        """Initialize an energy-resolved loss plotter."""
        super().__init__(plotter_type, latex_font=latex_font, plot_size=plot_size)
        self.y_scale = y_scale
        self.start_index = start_index
        self.end_index = end_index
        self.y_zoom = y_zoom
        self.keys = keys
        self.legend_position = legend_position

    def _plot(self, results: AnalysisResults, output_dir: Path) -> None:
        """Write per-method and combined energy-resolved loss curve PDFs.

        Args:
            results: Analysis pipeline outputs to plot.
            output_dir: Directory where the ``energy_resolved_loss_plots`` tree should be written.

        Raises:
            StopIteration: If no ``vector`` aggregator result exists.
        """
        root = output_dir / "energy_resolved_loss_plots"

        series: list[tuple[str, list[str], str, dict[str, Curve]]] = []
        key_order: list[str] = []

        for reader_idx, reader_selectors in enumerate(results.selectors):
            logging.info(f"    Predictions {reader_idx + 1}/{len(results.selectors)}.")

            for sel_idx in range(len(reader_selectors)):
                logging.info(f"      Selector {sel_idx + 1}/{len(reader_selectors)}.")
                label = results.method_label(reader_idx, sel_idx)

                data = results.aggregation(reader_idx, sel_idx, "vector").data
                curves = self._select_curves(data)
                if not curves:
                    continue

                color = method_color(len(series))
                series.append((label.dir_name, label.lines, color, curves))
                for key in curves:
                    if key not in key_order:
                        key_order.append(key)

        if not series:
            logging.info("    No energy-resolved loss data collected, skipping.")
            return

        for dir_name, label_lines, color, curves in series:
            combo_dir = root / dir_name
            combo_dir.mkdir(parents=True, exist_ok=True)

            for key, (mean, std) in curves.items():
                self._curve_figure(mean, std, key, one_line_label(label_lines), color, combo_dir / f"{key}.pdf")

        combined_dir = root / "combined"
        combined_dir.mkdir(parents=True, exist_ok=True)
        for key in key_order:
            key_series: list[tuple[str, list[str], Curve]] = [
                (color, label_lines, curves[key]) for _, label_lines, color, curves in series if key in curves
            ]
            self._combined_grid(key, key_series, combined_dir)
            self._combined_overlay(key, key_series, combined_dir)

    def _select_curves(self, data: dict[str, Any]) -> dict[str, Curve]:
        """Restrict aggregated per-channel curves to the configured keys and channels.

        Args:
            data: ``vector`` aggregator data mapping each key to its ``mean``
                and ``std`` curves.

        Returns:
            Mapping from loss key to ``(mean, std)`` curves with shape ``(N,)``.
        """
        channel_slice = slice(self.start_index, self.end_index)
        return {
            key: (entry["mean"][channel_slice], entry["std"][channel_slice])
            for key, entry in data.items()
            if self.keys is None or key in self.keys
        }

    def _apply_axis(self, ax: Axes) -> None:
        """Apply the configured y-axis scaling and zoom to one axis.

        Args:
            ax: Matplotlib axis to configure.
        """
        ax.set_yscale(self.y_scale)
        if self.y_scale == "log":
            positive_values = [
                value
                for line in ax.lines
                for value in np.asarray(line.get_ydata(), dtype=float).ravel()
                if np.isfinite(value) and value > 0
            ]
            upper = self.y_zoom if self.y_zoom is not None else ax.get_ylim()[1]
            if not np.isfinite(upper) or upper <= 0:
                upper = max(abs(upper) if np.isfinite(upper) else 0.0, 1e-12)
            lower = min(positive_values) * 0.5 if positive_values else max(upper * 1e-6, 1e-12)
            lower = min(max(lower, 1e-12), upper * 0.5)
            ax.set_ylim(lower, upper)
        elif self.y_zoom is not None:
            ax.set_ylim(0, self.y_zoom)
        apply_decimal_tick_format(ax)

    def _curve_figure(
        self,
        mean: np.ndarray,
        std: np.ndarray,
        key: str,
        subtitle: str,
        color: str,
        out: Path,
    ) -> None:
        """Write one mean-loss curve figure for a single method.

        Args:
            mean: Per-channel mean loss with shape ``(N,)``.
            std: Per-channel loss standard deviation with shape ``(N,)``.
            key: Loss key used for labels and output path context.
            subtitle: Plot subtitle text describing prediction and selector context.
            color: Method color used for the curve.
            out: Destination PDF path.
        """
        fig, ax = single_panel(self.style, "energy", legend_position=self.legend_position)
        x = np.arange(len(mean))
        ax.plot(x, mean, color=color, linewidth=self.style.linewidth("main"), label="Mean")
        ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.25, linewidth=0, label="+/- 1 std")
        ax.set_xlabel("Energy")
        ax.set_ylabel(key)
        add_legend(ax, self.style, position=self.legend_position)
        self._apply_axis(ax)
        set_context_title(ax, subtitle, self.style)
        save_figure(fig, out)

    def _combined_grid(
        self,
        key: str,
        series: list[tuple[str, list[str], Curve]],
        out: Path,
    ) -> None:
        """Write one combined figure with a per-method curve grid.

        All cells share one x range and one y range so the cells stay directly
        comparable.

        Args:
            key: Loss key used for labels and output path context.
            series: ``(color, label lines, (mean, std))`` per method in first-seen order.
            out: Directory where the combined grid PDF should be written.
        """
        n = len(series)

        fig, axes = method_grid(n, self.style, geometry="energy")
        ncols = len(axes[0])

        for idx, (color, label_lines, (mean, std)) in enumerate(series):
            ax = axes[idx // ncols][idx % ncols]
            x = np.arange(len(mean))
            ax.plot(x, mean, color=color, linewidth=self.style.linewidth("secondary"))
            ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.25, linewidth=0)
            style_grid_cell(ax, label_lines, self.style)

        finish_grid(axes, n, "Energy", key, self.style)
        self._apply_axis(axes[0][0])
        save_figure(fig, out / f"{key}_grid.pdf")

    def _combined_overlay(
        self,
        key: str,
        series: list[tuple[str, list[str], Curve]],
        out: Path,
    ) -> None:
        """Write one combined figure overlaying all mean-loss curves.

        Args:
            key: Loss key used for labels and output path context.
            series: ``(color, label lines, (mean, std))`` per method in first-seen order.
            out: Directory where the combined overlay PDF should be written.
        """
        fig, ax = single_panel(self.style, "energy", legend_position=self.legend_position)
        for color, label_lines, (mean, std) in series:
            x = np.arange(len(mean))
            ax.plot(
                x, mean, color=color, linewidth=self.style.linewidth("secondary"), label=one_line_label(label_lines)
            )
            ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.12, linewidth=0)
        ax.set_xlabel("Energy")
        ax.set_ylabel(key)
        add_legend(ax, self.style, position=self.legend_position)
        self._apply_axis(ax)
        save_figure(fig, out / f"{key}_overlay.pdf")

    @property
    def signature(self) -> Config:
        """Return the energy-resolved loss plotter signature.

        Returns:
            Configuration values needed to recreate this plotter.
        """
        signature = super().signature
        signature.update_with_dict(
            {
                "y_scale": self.y_scale,
                "start_index": self.start_index,
                "end_index": self.end_index,
                "y_zoom": self.y_zoom,
                "keys": self.keys,
                "legend_position": self.legend_position,
            }
        )
        return signature
