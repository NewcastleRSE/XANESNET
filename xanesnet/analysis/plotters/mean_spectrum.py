# SPDX-License-Identifier: GPL-3.0-or-later
#
# XANESNET
#
# Authors:  Hendrik Junkawitsch, Tom J. Penfold, Thomas J. Pope, C. D. Rankine, B. Li
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
#   Junkawitsch et al., "XANESNET: A Modular, Extensible, and Flexible Machine Learning Framework for Spectroscopy."
#   Rankine et al., "A Deep Neural Network for the Rapid Prediction of X-ray Absorption Spectra."
#   Rankine et al., "Accurate, affordable, and generalizable machine learning simulations of transition metal x-ray absorption spectra using the XANESNET deep neural network."
#   Penfold et al., "Field-Aware Energy-Conditioned Message Passing Neural Networks for Absorber-Centred Modelling of X-ray Spectroscopy."
#   Penfold et al., "A deep neural network for valence-to-core X-ray emission spectroscopy."
#   Falbo et al., "On the Analysis of X-ray Absorption Spectra for Polyoxometallates."
#   Madkhali et al., "Enhancing the Analysis of Disorder in X-ray Absorption Spectra: Application of Deep Neural Networks to T-Jump X-ray Probe Experiments."
#   Madkhali et al., "The Role of Structural Representation in the Performance of a Deep Neural Network for X-ray Spectroscopy."

"""Plotter for mean-spectrum regression diagnostics."""

import logging
from pathlib import Path

import numpy as np
from matplotlib.axes import Axes
from matplotlib.lines import Line2D

from xanesnet.serialization.config import Config

from ..result import AnalysisResults
from ..utils import one_line_label
from .base import Plotter
from .common.channels import CHANNEL_INDEX_LABEL, channel_indices, slice_channels
from .common.formatting import format_decimal
from .common.layout import save_figure, single_panel
from .common.style import (
    COLOR_ACCENT_GREEN,
    COLOR_ACCENT_RED,
    COLOR_TARGET,
    PlotSize,
    add_legend,
    method_color,
    set_context_title,
)
from .registry import PlotterRegistry

# Method mean statistics: label lines, color, mean target, mean prediction,
# target std, prediction std, best-percent prediction mean/std, worst-percent
# prediction mean/std.
_MethodMeans = tuple[
    list[str],
    str,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]


@PlotterRegistry.register("mean_spectrum")
class MeanSpectrumPlotter(Plotter):
    """Plot mean predicted and target spectra plus best/worst tail means.

    Writes ``mean_spectrum.pdf``, ``tail_means.pdf``, and ``spread.pdf`` per
    method, plus combined overlays.

    Requires:
        Spectrum statistics: provided by the ``spectrum`` aggregator.
        Best/worst tail statistics: provided by the ``ranking`` aggregator.

    Args:
        plotter_type: Registered plotter name from the analysis configuration.
        latex_font: Render figures in a LaTeX-style serif font when ``True``.
        start_index: First channel index to plot; ``None`` starts at channel zero.
        end_index: One-past-the-end channel index; ``None`` includes the remaining channels.
        legend_position: Place legends ``"inside"`` the axes or ``"outside"``
            them on the right.
        plot_size: Shared figure size profile: ``"small"`` or ``"default"``.
    """

    def __init__(
        self,
        plotter_type: str,
        start_index: int | None,
        end_index: int | None,
        legend_position: str,
        latex_font: bool,
        plot_size: PlotSize,
    ) -> None:
        """Initialize a mean-spectrum plotter."""
        super().__init__(plotter_type, latex_font=latex_font, plot_size=plot_size)
        self.start_index = start_index
        self.end_index = end_index
        self.legend_position = legend_position

    def _plot(self, results: AnalysisResults, output_dir: Path) -> None:
        """Write per-method mean-spectrum PDFs and combined overlays.

        Args:
            results: Analysis pipeline outputs to plot.
            output_dir: Directory where the ``mean_spectra`` tree should be written.

        Raises:
            StopIteration: If no ``spectrum`` or ``ranking`` aggregator result
                exists.
        """
        if not results.selectors:
            logging.info("    No selectors available, skipping.")
            return

        root = output_dir / "mean_spectra"
        root.mkdir(parents=True, exist_ok=True)

        methods: list[_MethodMeans] = []
        percent = 0.0

        for reader_idx, reader_selectors in enumerate(results.selectors):
            logging.info(f"    Predictions {reader_idx + 1}/{len(results.selectors)}.")

            for sel_idx in range(len(reader_selectors)):
                logging.info(f"      Selector {sel_idx + 1}/{len(reader_selectors)}.")
                label = results.method_label(reader_idx, sel_idx)

                spectrum_data = results.aggregation(reader_idx, sel_idx, "spectrum").data
                ranking_data = results.aggregation(reader_idx, sel_idx, "ranking").data
                if not spectrum_data or not ranking_data:
                    continue

                target_mean = slice_channels(spectrum_data["target_mean"], self.start_index, self.end_index)
                pred_mean = slice_channels(spectrum_data["prediction_mean"], self.start_index, self.end_index)
                target_std = slice_channels(spectrum_data["target_std"], self.start_index, self.end_index)
                pred_std = slice_channels(spectrum_data["prediction_std"], self.start_index, self.end_index)
                best_mean = slice_channels(ranking_data["best_prediction_mean"], self.start_index, self.end_index)
                best_std = slice_channels(ranking_data["best_prediction_std"], self.start_index, self.end_index)
                worst_mean = slice_channels(ranking_data["worst_prediction_mean"], self.start_index, self.end_index)
                worst_std = slice_channels(ranking_data["worst_prediction_std"], self.start_index, self.end_index)
                percent = float(ranking_data["percent"])

                color = method_color(len(methods))
                methods.append(
                    (
                        label.lines,
                        color,
                        target_mean,
                        pred_mean,
                        target_std,
                        pred_std,
                        best_mean,
                        best_std,
                        worst_mean,
                        worst_std,
                    )
                )

                combo_dir = root / label.dir_name
                combo_dir.mkdir(parents=True, exist_ok=True)
                subtitle = one_line_label(label.lines)

                self._mean_figure(target_mean, pred_mean, pred_std, subtitle, color, combo_dir / "mean_spectrum.pdf")
                self._tail_figure(
                    target_mean,
                    pred_mean,
                    best_mean,
                    best_std,
                    worst_mean,
                    worst_std,
                    percent,
                    subtitle,
                    combo_dir / "tail_means.pdf",
                )
                self._spread_figure(target_std, pred_std, subtitle, color, combo_dir / "spread.pdf")

        if not methods:
            logging.info("    No samples selected, skipping.")
            return

        combined_dir = root / "combined"
        combined_dir.mkdir(parents=True, exist_ok=True)
        self._combined_mean_overlay(methods, combined_dir / "mean_spectrum_overlay.pdf")
        self._combined_tail_overlay(methods, percent, combined_dir / "tail_best_overlay.pdf", "best")
        self._combined_tail_overlay(methods, percent, combined_dir / "tail_worst_overlay.pdf", "worst")

    def _tail_figure(
        self,
        target_mean: np.ndarray,
        pred_mean: np.ndarray,
        best_mean: np.ndarray,
        best_std: np.ndarray,
        worst_mean: np.ndarray,
        worst_std: np.ndarray,
        percent: float,
        subtitle: str,
        out: Path,
    ) -> None:
        """Write one figure comparing the mean prediction with its best/worst tails.

        Args:
            target_mean: Per-channel mean target spectrum.
            pred_mean: Per-channel mean predicted spectrum.
            best_mean: Per-channel mean prediction of the best ``percent`` samples.
            best_std: Per-channel prediction std of the best ``percent`` samples.
            worst_mean: Per-channel mean prediction of the worst ``percent`` samples.
            worst_std: Per-channel prediction std of the worst ``percent`` samples.
            percent: Tail size in percent, as configured on the aggregator.
            subtitle: Plot subtitle text describing prediction and selector context.
            out: Destination PDF path.
        """
        fig, ax = single_panel(self.style, "energy", legend_position=self.legend_position)
        x = channel_indices(pred_mean)
        ax.plot(
            x,
            target_mean,
            color=COLOR_TARGET,
            linewidth=self.style.linewidth("light"),
            linestyle=":",
            label="Target mean",
        )
        ax.plot(x, pred_mean, color="black", linewidth=self.style.linewidth("main"), label="Total prediction mean")
        ax.fill_between(
            x,
            best_mean - best_std,
            best_mean + best_std,
            color=COLOR_ACCENT_GREEN,
            alpha=0.15,
            linewidth=0,
            label=f"Best {format_decimal(percent, 4)}% +/- 1 std",
        )
        ax.plot(
            x,
            best_mean,
            color=COLOR_ACCENT_GREEN,
            linewidth=self.style.linewidth("emphasis"),
            label=f"Best {format_decimal(percent, 4)}% mean",
        )
        ax.fill_between(
            x,
            worst_mean - worst_std,
            worst_mean + worst_std,
            color=COLOR_ACCENT_RED,
            alpha=0.15,
            linewidth=0,
            label=f"Worst {format_decimal(percent, 4)}% +/- 1 std",
        )
        ax.plot(
            x,
            worst_mean,
            color=COLOR_ACCENT_RED,
            linewidth=self.style.linewidth("emphasis"),
            label=f"Worst {format_decimal(percent, 4)}% mean",
        )
        ax.set_xlabel(CHANNEL_INDEX_LABEL)
        ax.set_ylabel("Intensity")
        add_legend(ax, self.style, position=self.legend_position)
        set_context_title(ax, subtitle, self.style)
        save_figure(fig, out)

    def _mean_figure(
        self,
        target_mean: np.ndarray,
        pred_mean: np.ndarray,
        pred_std: np.ndarray,
        subtitle: str,
        color: str,
        out: Path,
    ) -> None:
        """Write one figure comparing the mean prediction and mean target spectra.

        Args:
            target_mean: Per-channel mean target spectrum.
            pred_mean: Per-channel mean predicted spectrum.
            pred_std: Per-channel prediction standard deviation.
            subtitle: Plot subtitle text describing prediction and selector context.
            color: Method color used for the prediction uncertainty band.
            out: Destination PDF path.
        """
        fig, ax = single_panel(self.style, "energy", legend_position=self.legend_position)
        x = channel_indices(pred_mean)
        ax.plot(x, target_mean, color=COLOR_TARGET, linewidth=self.style.linewidth("secondary"), label="Target mean")
        ax.fill_between(
            x,
            pred_mean - pred_std,
            pred_mean + pred_std,
            color=color,
            alpha=0.25,
            linewidth=0,
            label="Prediction +/- 1 std",
        )
        ax.plot(x, pred_mean, color="black", linewidth=self.style.linewidth("main"), label="Prediction mean")
        ax.set_xlabel(CHANNEL_INDEX_LABEL)
        ax.set_ylabel("Intensity")
        add_legend(ax, self.style, position=self.legend_position)
        set_context_title(ax, subtitle, self.style)
        save_figure(fig, out)

    def _spread_figure(
        self,
        target_std: np.ndarray,
        pred_std: np.ndarray,
        subtitle: str,
        color: str,
        out: Path,
    ) -> None:
        """Write one figure comparing per-channel prediction and target spread.

        A prediction spread consistently below the target spread indicates
        that the model predicts the mean spectrum.

        Args:
            target_std: Per-channel target standard deviation.
            pred_std: Per-channel prediction standard deviation.
            subtitle: Plot subtitle text describing prediction and selector context.
            color: Method color used for the prediction spread curve.
            out: Destination PDF path.
        """
        fig, ax = single_panel(self.style, "energy", legend_position=self.legend_position)
        x = channel_indices(pred_std)
        ax.plot(x, target_std, color=COLOR_TARGET, linewidth=self.style.linewidth("emphasis"), label="Target std")
        ax.plot(x, pred_std, color=color, linewidth=self.style.linewidth("emphasis"), label="Prediction std")
        ax.set_xlabel(CHANNEL_INDEX_LABEL)
        ax.set_ylabel("Standard deviation")
        add_legend(ax, self.style, position=self.legend_position)
        set_context_title(ax, subtitle, self.style)
        save_figure(fig, out)

    def _legend_with_styles(self, ax: Axes, extra: list[tuple[str, str]]) -> None:
        """Rebuild the axis legend with gray style entries appended.

        Args:
            ax: Matplotlib axis whose legend should be extended.
            extra: ``(label, linestyle)`` pairs to append as gray proxy entries.
        """
        handles, labels = ax.get_legend_handles_labels()
        existing = set(labels)
        for label, linestyle in extra:
            if label in existing:
                continue
            handles.append(
                Line2D(
                    [0],
                    [0],
                    color="gray",
                    linewidth=self.style.linewidth("light"),
                    linestyle=linestyle,
                    label=label,
                )
            )
            labels.append(label)
        add_legend(ax, self.style, position=self.legend_position, handles=handles, labels=labels)

    @staticmethod
    def _shared_target_mean(methods: list[_MethodMeans]) -> np.ndarray | None:
        """Return the common target mean when all methods share one.

        Args:
            methods: Method mean statistics in first-seen order.

        Returns:
            The shared per-channel target mean, or ``None`` when the methods
            have different target means.
        """
        first = methods[0][2]
        if all(np.allclose(first, method[2]) for method in methods[1:]):
            return first
        return None

    def _combined_mean_overlay(self, methods: list[_MethodMeans], out: Path) -> None:
        """Write one combined figure overlaying all mean prediction/target spectra.

        When all methods share the same target mean it is drawn once as a
        gray dotted line; otherwise one dotted target curve per method is
        drawn in the method color.

        Args:
            methods: Method mean statistics in first-seen order.
            out: Destination PDF path.
        """
        fig, ax = single_panel(self.style, "energy", legend_position=self.legend_position)
        shared_target = self._shared_target_mean(methods)
        if shared_target is not None:
            ax.plot(
                channel_indices(shared_target),
                shared_target,
                color=COLOR_TARGET,
                linewidth=self.style.linewidth("secondary"),
                linestyle=":",
                label="Target mean",
            )
        for label_lines, color, target_mean, pred_mean, *_ in methods:
            x = channel_indices(pred_mean)
            if shared_target is None:
                ax.plot(x, target_mean, color=color, linewidth=self.style.linewidth("light"), linestyle=":")
            ax.plot(
                x, pred_mean, color=color, linewidth=self.style.linewidth("main"), label=one_line_label(label_lines)
            )
        ax.set_xlabel(CHANNEL_INDEX_LABEL)
        ax.set_ylabel("Intensity")
        self._legend_with_styles(ax, [("Prediction mean", "-"), ("Target mean", ":")])
        save_figure(fig, out)

    def _combined_tail_overlay(self, methods: list[_MethodMeans], percent: float, out: Path, direction: str) -> None:
        """Write one combined figure overlaying per-method tail, prediction, and target means.

        When all methods share the same target mean it is drawn once as a
        gray dotted line; otherwise one dotted target curve per method is
        drawn in the method color.

        Args:
            methods: Method mean statistics in first-seen order.
            percent: Tail size in percent, as configured on the aggregator.
            out: Destination PDF path.
            direction: ``"best"`` or ``"worst"``, selecting which tail mean is
                drawn as the solid per-method curve.
        """
        fig, ax = single_panel(self.style, "energy", legend_position=self.legend_position)
        shared_target = self._shared_target_mean(methods)
        if shared_target is not None:
            ax.plot(
                channel_indices(shared_target),
                shared_target,
                color=COLOR_TARGET,
                linewidth=self.style.linewidth("secondary"),
                linestyle=":",
                label="Target mean",
            )
        for label_lines, color, target_mean, pred_mean, _, _, best_mean, _, worst_mean, _ in methods:
            tail_mean = best_mean if direction == "best" else worst_mean
            x = channel_indices(tail_mean)
            if shared_target is None:
                ax.plot(x, target_mean, color=color, linewidth=self.style.linewidth("light"), linestyle=":")
            ax.plot(x, pred_mean, color=color, linewidth=self.style.linewidth("secondary"), linestyle="--")
            ax.plot(
                x, tail_mean, color=color, linewidth=self.style.linewidth("main"), label=one_line_label(label_lines)
            )
        ax.set_xlabel(CHANNEL_INDEX_LABEL)
        ax.set_ylabel("Intensity")
        self._legend_with_styles(
            ax,
            [
                (f"{direction.capitalize()} {format_decimal(percent, 4)}% mean", "-"),
                ("Total prediction mean", "--"),
                ("Target mean", ":"),
            ],
        )
        save_figure(fig, out)

    @property
    def signature(self) -> Config:
        """Return the mean-spectrum plotter signature.

        Returns:
            Configuration values needed to recreate this plotter.
        """
        signature = super().signature
        signature.update_with_dict(
            {
                "start_index": self.start_index,
                "end_index": self.end_index,
                "legend_position": self.legend_position,
            }
        )
        return signature
