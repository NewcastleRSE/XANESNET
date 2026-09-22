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

"""Plotter for the per-channel bias-variance error decomposition."""

import logging
from pathlib import Path

import numpy as np
from matplotlib.axes import Axes

from xanesnet.serialization.config import Config

from ..result import AnalysisResults
from ..utils import one_line_label
from .base import Plotter
from .common.layout import (
    finish_grid,
    method_grid,
    save_figure,
    single_panel,
    style_grid_cell,
)
from .common.style import (
    COLOR_ACCENT_GREEN,
    COLOR_ACCENT_RED,
    PlotSize,
    add_figure_legend,
    add_legend,
    method_color,
    set_context_title,
)
from .registry import PlotterRegistry

# Label lines, color, MSE, squared bias, variance.
_MethodStats = tuple[list[str], str, np.ndarray, np.ndarray, np.ndarray]

_BIAS_COLOR = COLOR_ACCENT_RED
_VARIANCE_COLOR = COLOR_ACCENT_GREEN


@PlotterRegistry.register("bias_variance")
class BiasVariancePlotter(Plotter):
    """Plot the per-channel bias-variance decomposition of the MSE per method.

    A combined grid compares all methods on shared axes.

    Requires:
        Per-channel bias/variance decomposition: provided by the ``bias_variance`` aggregator.

    Args:
        plotter_type: Registered plotter name from the analysis configuration.
        latex_font: Render figures in a LaTeX-style serif font when ``True``.
        legend_position: Place the per-method legend ``"inside"`` the axes or
            ``"outside"`` it on the right. The combined grid keeps one shared
            legend above its cells.
        plot_size: Shared figure size profile: ``"small"`` or ``"default"``.
    """

    def __init__(
        self,
        plotter_type: str,
        legend_position: str,
        latex_font: bool,
        plot_size: PlotSize,
    ) -> None:
        """Initialize a bias-variance plotter."""
        super().__init__(plotter_type, latex_font=latex_font, plot_size=plot_size)
        self.legend_position = legend_position

    def _plot(self, results: AnalysisResults, output_dir: Path) -> None:
        """Write per-method decomposition PDFs and a combined grid.

        Args:
            results: Analysis pipeline outputs to plot.
            output_dir: Directory where the ``bias_variance_plots`` tree should be written.

        Raises:
            StopIteration: If no ``bias_variance`` aggregator result exists.
        """
        if not results.selectors:
            logging.info("    No selectors available, skipping.")
            return

        root = output_dir / "bias_variance_plots"
        root.mkdir(parents=True, exist_ok=True)

        methods: list[_MethodStats] = []

        for reader_idx, reader_selectors in enumerate(results.selectors):
            logging.info(f"    Predictions {reader_idx + 1}/{len(results.selectors)}.")

            for sel_idx in range(len(reader_selectors)):
                logging.info(f"      Selector {sel_idx + 1}/{len(reader_selectors)}.")
                label = results.method_label(reader_idx, sel_idx)

                data = results.aggregation(reader_idx, sel_idx, "bias_variance").data
                if not data:
                    continue
                stats = (data["mse"], data["bias2"], data["variance"])

                color = method_color(len(methods))
                methods.append((label.lines, color, *stats))

                combo_dir = root / label.dir_name
                combo_dir.mkdir(parents=True, exist_ok=True)
                self._stats_figure(*stats, one_line_label(label.lines), color, combo_dir / "bias_variance.pdf")

        if not methods:
            logging.info("    No samples selected, skipping.")
            return

        combined_dir = root / "combined"
        combined_dir.mkdir(parents=True, exist_ok=True)
        self._stats_grid(methods, combined_dir / "bias_variance_grid.pdf")

    def _draw_decomposition(
        self, ax: Axes, mse: np.ndarray, bias2: np.ndarray, variance: np.ndarray, color: str
    ) -> None:
        """Draw the three decomposition curves into one axis.

        Args:
            ax: Matplotlib axis to draw on.
            mse: Per-channel mean squared error with shape ``(N,)``.
            bias2: Per-channel squared bias with shape ``(N,)``.
            variance: Per-channel variance with shape ``(N,)``.
            color: Method color used for the MSE curve.
        """
        x = np.arange(len(mse))
        ax.plot(x, mse, color=color, linewidth=self.style.linewidth("main"), label="MSE")
        ax.plot(
            x,
            bias2,
            color=_BIAS_COLOR,
            linewidth=self.style.linewidth("secondary"),
            linestyle="--",
            label="Bias^2",
        )
        ax.plot(
            x,
            variance,
            color=_VARIANCE_COLOR,
            linewidth=self.style.linewidth("secondary"),
            linestyle=":",
            label="Variance",
        )

    def _stats_figure(
        self,
        mse: np.ndarray,
        bias2: np.ndarray,
        variance: np.ndarray,
        subtitle: str,
        color: str,
        out: Path,
    ) -> None:
        """Write one decomposition figure for a single method.

        Args:
            mse: Per-channel mean squared error with shape ``(N,)``.
            bias2: Per-channel squared bias with shape ``(N,)``.
            variance: Per-channel variance with shape ``(N,)``.
            subtitle: Plot subtitle text describing prediction and selector context.
            color: Method color used for the MSE curve.
            out: Destination PDF path.
        """
        fig, ax = single_panel(self.style, "energy", legend_position=self.legend_position)
        self._draw_decomposition(ax, mse, bias2, variance, color)
        ax.set_xlabel("Energy")
        ax.set_ylabel("Loss")
        add_legend(ax, self.style, position=self.legend_position)
        set_context_title(ax, subtitle, self.style)
        save_figure(fig, out)

    def _stats_grid(self, methods: list[_MethodStats], out: Path) -> None:
        """Write one combined figure with a per-method decomposition grid.

        All cells share one x range and one y range so the cells stay directly
        comparable.

        Args:
            methods: Method decomposition statistics in first-seen order.
            out: Destination PDF path.
        """
        n = len(methods)

        fig, axes = method_grid(n, self.style, geometry="energy", height_margin=0.55)
        ncols = len(axes[0])

        for idx, (label_lines, color, mse, bias2, variance) in enumerate(methods):
            ax = axes[idx // ncols][idx % ncols]
            self._draw_decomposition(ax, mse, bias2, variance, color)
            style_grid_cell(ax, label_lines, self.style)

        finish_grid(axes, n, "Energy", "Loss", self.style)
        handles, labels = axes[0][0].get_legend_handles_labels()
        add_figure_legend(fig, handles, labels, self.style, ncol=3, bbox_to_anchor=(0.5, 0.98))
        save_figure(fig, out)

    @property
    def signature(self) -> Config:
        """Return the bias-variance plotter signature.

        Returns:
            Configuration values needed to recreate this plotter.
        """
        signature = super().signature
        signature.update_with_dict({"legend_position": self.legend_position})
        return signature
