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

"""Plotter for predicted-versus-target intensity parity plots."""

import logging
from pathlib import Path

import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import LinearSegmentedColormap, LogNorm

from ..result import AnalysisResults
from ..selectors import Selector
from ..utils import one_line_label
from .base import Plotter
from .common.formatting import format_decimal
from .common.layout import (
    add_colorbar,
    finish_grid,
    method_grid,
    save_figure,
    single_panel,
    style_grid_cell,
)
from .common.style import PlotSize, add_note, method_color, set_context_title
from .registry import PlotterRegistry

# Label lines, color, flattened targets, flattened predictions.
_MethodPoints = tuple[list[str], str, np.ndarray, np.ndarray]


@PlotterRegistry.register("parity")
class ParityPlotter(Plotter):
    """Plot predicted-versus-target intensity parity per method and combined.

    All selected sample spectra are flattened into ``(target, prediction)``
    intensity pairs and drawn as a density scatter around the identity line,
    annotated with RMSE, MAE, and R2. A combined grid compares one panel per
    method.

    Args:
        plotter_type: Registered plotter name from the analysis configuration.
        latex_font: Render figures in a LaTeX-style serif font when ``True``.
        plot_size: Shared figure size profile: ``"small"`` or ``"default"``.
    """

    def __init__(self, plotter_type: str, latex_font: bool, plot_size: PlotSize) -> None:
        """Initialize a parity plotter."""
        super().__init__(plotter_type, latex_font=latex_font, plot_size=plot_size)

    def _plot(self, results: AnalysisResults, output_dir: Path) -> None:
        """Write per-method parity PDFs and a combined parity grid.

        Args:
            results: Analysis pipeline outputs to plot.
            output_dir: Directory where the ``parity_plots`` tree should be written.
        """
        if not results.selectors:
            logging.info("    No selectors available, skipping.")
            return

        root = output_dir / "parity_plots"
        root.mkdir(parents=True, exist_ok=True)

        methods: list[_MethodPoints] = []

        for reader_idx, reader_selectors in enumerate(results.selectors):
            logging.info(f"    Predictions {reader_idx + 1}/{len(results.selectors)}.")

            for sel_idx, selector in enumerate(reader_selectors):
                logging.info(f"      Selector {sel_idx + 1}/{len(reader_selectors)}.")
                label = results.method_label(reader_idx, sel_idx)

                points = self._collect_points(selector)
                if points is None:
                    continue

                targets, preds = points
                color = method_color(len(methods))
                methods.append((label.lines, color, targets, preds))

                combo_dir = root / label.dir_name
                combo_dir.mkdir(parents=True, exist_ok=True)
                self._parity_figure(targets, preds, one_line_label(label.lines), color, combo_dir / "parity.pdf")

        if not methods:
            logging.info("    No samples selected, skipping.")
            return

        combined_dir = root / "combined"
        combined_dir.mkdir(parents=True, exist_ok=True)
        self._parity_grid(methods, combined_dir / "parity_grid.pdf")

    @staticmethod
    def _collect_points(selector: Selector) -> tuple[np.ndarray, np.ndarray] | None:
        """Flatten all selected spectra into target and prediction intensity vectors.

        Args:
            selector: Selector over prediction samples for one prediction reader and selector pair.

        Returns:
            ``(targets, preds)`` flattened vectors with shape ``(M,)``, or
            ``None`` when no samples are selected.
        """
        targets: list[np.ndarray] = []
        preds: list[np.ndarray] = []
        for sample in selector:
            targets.append(np.asarray(sample["target"]).ravel())
            preds.append(np.asarray(sample["prediction"]).ravel())
        if not targets:
            return None
        return np.concatenate(targets), np.concatenate(preds)

    @staticmethod
    def _parity_metrics(targets: np.ndarray, preds: np.ndarray) -> tuple[float, float, float | None]:
        """Compute RMSE, MAE, and R2 over flattened intensity pairs.

        Args:
            targets: Flattened target intensities with shape ``(M,)``.
            preds: Flattened predicted intensities with shape ``(M,)``.

        Returns:
            ``(rmse, mae, r2)`` where ``r2`` is ``None`` when the target
            intensities are constant.
        """
        diff = preds - targets
        rmse = float(np.sqrt(np.mean(diff**2)))
        mae = float(np.mean(np.abs(diff)))
        ss_tot = float(np.sum((targets - targets.mean()) ** 2))
        r2 = 1.0 - float(np.sum(diff**2)) / ss_tot if ss_tot > 0 else None
        return rmse, mae, r2

    def _parity_figure(
        self,
        targets: np.ndarray,
        preds: np.ndarray,
        subtitle: str,
        color: str,
        out: Path,
    ) -> None:
        """Write one parity figure for a single method.

        Args:
            targets: Flattened target intensities with shape ``(M,)``.
            preds: Flattened predicted intensities with shape ``(M,)``.
            subtitle: Plot subtitle text describing prediction and selector context.
            color: Method color used for the density colormap.
            out: Destination PDF path.
        """
        fig, ax = single_panel(self.style, "square", extra_bottom=0.45)
        self._draw_parity_panel(ax, targets, preds, color, gridsize=80)
        add_colorbar(fig, ax.collections[-1], self.style, orientation="horizontal", label="")
        ax.set_xlabel("Target intensity")
        ax.set_ylabel("Predicted intensity")
        self._add_metrics_text(ax, targets, preds)
        set_context_title(ax, subtitle, self.style)
        save_figure(fig, out)

    def _draw_parity_panel(self, ax: Axes, targets: np.ndarray, preds: np.ndarray, color: str, gridsize: int) -> None:
        """Draw one parity density panel with its identity line.

        Args:
            ax: Matplotlib axis to draw on.
            targets: Flattened target intensities with shape ``(M,)``.
            preds: Flattened predicted intensities with shape ``(M,)``.
            color: Method color used for the density colormap.
            gridsize: Number of hexagons along each axis.
        """
        cmap = LinearSegmentedColormap.from_list("parity_density", ["#ffffff", color])
        poly = ax.hexbin(targets, preds, gridsize=gridsize, mincnt=1, cmap=cmap)
        counts = poly.get_array()
        if counts is not None and counts.size:
            compressed = getattr(counts, "compressed", None)
            data = compressed() if compressed is not None else np.asarray(counts)
            vmax = float(np.max(data)) if data.size else 2.0
        else:
            vmax = 2.0
        poly.set_norm(LogNorm(vmin=1.0, vmax=max(2.0, vmax)))
        lo = min(float(targets.min()), float(preds.min()))
        hi = max(float(targets.max()), float(preds.max()))
        ax.plot(
            [lo, hi],
            [lo, hi],
            color="black",
            linewidth=self.style.linewidth("identity"),
            linestyle="--",
        )
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        # Keep the axes box square so the identity line stays at 45 degrees
        # and the density cloud is not stretched horizontally.
        ax.set_aspect("equal")

    def _add_metrics_text(self, ax: Axes, targets: np.ndarray, preds: np.ndarray) -> None:
        """Annotate a parity panel with the flattened-pair metrics.

        Args:
            ax: Matplotlib axis to annotate.
            targets: Flattened target intensities with shape ``(M,)``.
            preds: Flattened predicted intensities with shape ``(M,)``.
        """
        rmse, mae, r2 = self._parity_metrics(targets, preds)
        r2_text = format_decimal(r2, 4) if r2 is not None else "n/a"
        text = f"n={len(targets)}\nRMSE={format_decimal(rmse, 4)}" f"\nMAE={format_decimal(mae, 4)}\nR2={r2_text}"
        add_note(ax, text, self.style, location="lower right")

    def _parity_grid(self, methods: list[_MethodPoints], out: Path) -> None:
        """Write one combined figure with a per-method parity density grid.

        All cells share one x range and one y range so the cells stay directly
        comparable.

        Args:
            methods: Method intensity pairs in first-seen order.
            out: Destination PDF path.
        """
        n = len(methods)

        # Square cells: each parity panel enforces equal x and y ranges and an
        # equal aspect, so the cell size must match to avoid skewed density
        # clouds and a stretched identity line. The manual adjust keeps the
        # cells as large and close together as possible.
        fig, axes = method_grid(
            n,
            self.style,
            geometry="square",
            width_margin=0.45,
            height_margin=0.55,
        )
        ncols = len(axes[0])

        for idx, (label_lines, color, targets, preds) in enumerate(methods):
            ax = axes[idx // ncols][idx % ncols]
            self._draw_parity_panel(ax, targets, preds, color, gridsize=40)
            rmse, _, _ = self._parity_metrics(targets, preds)
            add_note(ax, f"RMSE={format_decimal(rmse, 3)}", self.style, location="upper left")
            style_grid_cell(ax, label_lines, self.style)

        finish_grid(axes, n, "Target intensity", "Predicted intensity", self.style)
        vmax = 2.0
        for row in axes:
            for ax in row:
                for coll in ax.collections:
                    arr = coll.get_array()
                    if arr is None or not arr.size:
                        continue
                    compressed = getattr(arr, "compressed", None)
                    data = compressed() if compressed is not None else np.asarray(arr)
                    if data.size:
                        vmax = max(vmax, float(np.max(data)))
        norm = LogNorm(vmin=1.0, vmax=vmax)
        for row in axes:
            for ax in row:
                for coll in ax.collections:
                    coll.set_norm(norm)

        fig.canvas.draw()
        grid_right = max(ax.get_position().x1 for row in axes for ax in row)
        grid_bottom = min(ax.get_position().y0 for row in axes for ax in row)
        grid_top = max(ax.get_position().y1 for row in axes for ax in row)
        cbar_gap = self.style.grid_gaps()[0] / fig.get_figwidth()
        cbar_width = self.style.grid_margins()[1] / fig.get_figwidth()
        add_colorbar(
            fig,
            axes[0][0].collections[0],
            self.style,
            rect=(grid_right + cbar_gap, grid_bottom, cbar_width, grid_top - grid_bottom),
            label="count",
        )
        save_figure(fig, out)
