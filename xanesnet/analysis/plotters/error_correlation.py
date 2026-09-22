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

"""Plotter for cross-method per-sample error correlation."""

import logging
from itertools import repeat
from pathlib import Path

import numpy as np
from matplotlib.axes import Axes

from xanesnet.serialization.config import Config
from xanesnet.serialization.jsonl_stream import JSONLStream
from xanesnet.utils.exceptions import ConfigError

from ..result import AnalysisResults
from ..selectors import Selector
from ..utils import (
    SampleKey,
    is_scalar_value,
    one_line_label,
    sample_key,
    sample_key_sort_key,
)
from .base import Plotter
from .common.layout import add_grid_label, matrix_grid, save_figure, single_panel
from .common.style import PlotSize, add_note, method_color
from .registry import PlotterRegistry

# Label lines, color, compound sample identity to per-sample error.
_MethodErrors = tuple[list[str], str, dict[SampleKey, float]]


@PlotterRegistry.register("error_correlation")
class ErrorCorrelationPlotter(Plotter):
    """Plot pairwise per-sample error correlations across methods.

    One grid figure draws the pairwise error scatter on the samples shared by
    each method pair, annotated with the Pearson correlation and the number of
    common samples; one standalone figure per pair shows the same scatter.

    Requires:
        Per-sample error values: provided by a scalar collector emitting ``sort_key``.

    Args:
        plotter_type: Registered plotter name from the analysis configuration.
        latex_font: Render figures in a LaTeX-style serif font when ``True``.
        sort_key: Scalar collector key used as the per-sample error. Must be
            produced by a configured collector.
        plot_size: Shared figure size profile: ``"small"`` or ``"default"``.
    """

    def __init__(self, plotter_type: str, sort_key: str, latex_font: bool, plot_size: PlotSize) -> None:
        """Initialize an error correlation plotter."""
        super().__init__(plotter_type, latex_font=latex_font, plot_size=plot_size)
        self.sort_key = sort_key

    def _plot(self, results: AnalysisResults, output_dir: Path) -> None:
        """Write the pairwise error correlation grid and per-pair figures.

        Args:
            results: Analysis pipeline outputs to plot.
            output_dir: Directory where the ``error_correlation`` tree should be written.
        """
        if not results.selectors:
            logging.info("    No selectors available, skipping.")
            return

        methods: list[_MethodErrors] = []

        for reader_idx, reader_selectors in enumerate(results.selectors):
            logging.info(f"    Predictions {reader_idx + 1}/{len(results.selectors)}.")

            for sel_idx, selector in enumerate(reader_selectors):
                logging.info(f"      Selector {sel_idx + 1}/{len(reader_selectors)}.")
                label = results.method_label(reader_idx, sel_idx)
                stream = results.collector_stream(reader_idx, sel_idx)

                errors = self._collect_errors(selector, stream)
                if not errors:
                    continue

                methods.append((label.lines, method_color(len(methods)), errors))

        if len(methods) < 2:
            logging.info("    Need at least two methods for error correlation, skipping.")
            return

        root = output_dir / "error_correlation"
        root.mkdir(parents=True, exist_ok=True)
        self._correlation_grid(methods, root / "correlation_grid.pdf")

        for i in range(len(methods)):
            for j in range(i):
                pair = self._pair_values(methods[i][2], methods[j][2])
                if pair is None:
                    continue
                xs, ys = pair
                self._pair_figure(
                    xs,
                    ys,
                    methods[i][0],
                    methods[j][0],
                    methods[i][1],
                    root / f"pair_{i:03d}__{j:03d}.pdf",
                )

    def _collect_errors(self, selector: Selector, stream: JSONLStream | None) -> dict[SampleKey, float]:
        """Collect one method's errors keyed by compound sample identity.

        Args:
            selector: Selector over prediction samples for one prediction reader and selector pair.
            stream: Optional collector result stream aligned with ``selector``.

        Returns:
            Mapping from ``(sample_id, target_site_index)`` to the error value.

        Raises:
            ConfigError: If ``sort_key`` is missing or non-scalar, or if a
                compound sample identity occurs more than once.
        """
        errors: dict[SampleKey, float] = {}
        for sample, record in zip(selector, stream if stream is not None else repeat({})):
            value = record.get(self.sort_key)
            if not is_scalar_value(value):
                raise ConfigError(
                    f"Sort key '{self.sort_key}' is missing or not a scalar for sample "
                    f"'{sample['sample_id']}'. Configure a scalar collector that produces this key."
                )
            identity = sample_key(sample)
            if identity in errors:
                raise ConfigError(f"Duplicate prediction record for sample identity {identity!r}.")
            errors[identity] = float(value)
        return errors

    def _correlation_grid(self, methods: list[_MethodErrors], out: Path) -> None:
        """Write one grid figure with pairwise error correlation panels.

        Diagonal cells carry the method labels; lower-triangle cells compare
        the per-sample errors of the corresponding method pair.

        Args:
            methods: Method errors in first-seen order.
            out: Destination PDF path.
        """
        n = len(methods)

        fig, axes = matrix_grid(
            n,
            n,
            self.style,
            geometry="square",
            width_margin=0.8,
            height_margin=0.8,
        )
        for i in range(n):
            for j in range(n):
                ax = axes[i][j]
                if i == j:
                    add_grid_label(ax, methods[i][0], self.style)
                    continue
                if j > i:
                    ax.axis("off")
                    continue

                pair = ErrorCorrelationPlotter._pair_values(methods[i][2], methods[j][2])
                if pair is None:
                    ax.axis("off")
                    add_grid_label(ax, ["no common samples"], self.style, color="gray")
                    continue

                xs, ys = pair
                self._draw_pair_panel(ax, xs, ys, methods[i][1])

        for j in range(n):
            axes[n - 1][j].set_xlabel(
                "error",
                fontsize=self.style.fontsize("axis_label"),
                labelpad=self.style.spacing("label_pad"),
            )
        for i in range(n):
            axes[i][0].set_ylabel(
                "error",
                fontsize=self.style.fontsize("axis_label"),
                labelpad=self.style.spacing("label_pad"),
            )

        save_figure(fig, out)

    @staticmethod
    def _pair_values(
        errors_i: dict[SampleKey, float], errors_j: dict[SampleKey, float]
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """Return error arrays for records shared by two methods.

        Args:
            errors_i: Per-record errors of the row method.
            errors_j: Per-record errors of the column method.

        Returns:
            ``(xs, ys)`` error arrays for the common samples, or ``None`` when
            fewer than two samples are shared.
        """
        common = sorted(set(errors_i) & set(errors_j), key=sample_key_sort_key)
        if len(common) < 2:
            return None
        return np.array([errors_i[s] for s in common]), np.array([errors_j[s] for s in common])

    def _draw_pair_panel(self, ax: Axes, xs: np.ndarray, ys: np.ndarray, color: str) -> None:
        """Draw one pairwise error scatter with identity line and annotation.

        Args:
            ax: Matplotlib axis to draw on.
            xs: Error values of the row method with shape ``(M,)``.
            ys: Error values of the column method with shape ``(M,)``.
            color: Method color used for the scatter.
        """
        lo = min(float(xs.min()), float(ys.min()))
        hi = max(float(xs.max()), float(ys.max()))
        ax.plot(
            [lo, hi],
            [lo, hi],
            color="black",
            linewidth=self.style.linewidth("grid_reference"),
            linestyle="--",
        )
        ax.scatter(xs, ys, s=self.style.scatter_size("error_correlation"), alpha=0.5, color=color)
        r = _pearson(xs, ys)
        note = f"r={r:.2f}" if r is not None else "r=n/a"
        add_note(ax, f"{note}\nn={len(xs)}", self.style, location="upper left")

    def _pair_figure(
        self,
        xs: np.ndarray,
        ys: np.ndarray,
        row_label_lines: list[str],
        col_label_lines: list[str],
        color: str,
        out: Path,
    ) -> None:
        """Write one standalone correlation figure for a single method pair.

        Args:
            xs: Error values of the row method with shape ``(M,)``.
            ys: Error values of the column method with shape ``(M,)``.
            row_label_lines: Label lines of the row method.
            col_label_lines: Label lines of the column method.
            color: Method color used for the scatter.
            out: Destination PDF path.
        """
        fig, ax = single_panel(self.style, "square")
        self._draw_pair_panel(ax, xs, ys, color)
        ax.set_xlabel(one_line_label(row_label_lines))
        ax.set_ylabel(one_line_label(col_label_lines))
        save_figure(fig, out)

    @property
    def signature(self) -> Config:
        """Return the error correlation plotter signature.

        Returns:
            Configuration values needed to recreate this plotter.
        """
        signature = super().signature
        signature.update_with_dict({"sort_key": self.sort_key})
        return signature


def _pearson(xs: np.ndarray, ys: np.ndarray) -> float | None:
    """Compute the Pearson correlation coefficient of two error vectors.

    Args:
        xs: Error values of the first method with shape ``(M,)``.
        ys: Error values of the second method with shape ``(M,)``.

    Returns:
        Correlation coefficient in ``[-1, 1]``, or ``None`` when either
        vector is constant.
    """
    xc = xs - xs.mean()
    yc = ys - ys.mean()
    denom = float(np.sqrt((xc**2).sum() * (yc**2).sum()))
    if denom == 0:
        return None
    return float((xc * yc).sum() / denom)
