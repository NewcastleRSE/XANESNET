# SPDX-License-Identifier: GPL-3.0-or-later
#
# XANESNET
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

"""Plotter for scalar value distributions."""

import logging
from pathlib import Path
from typing import Any, ClassVar

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from xanesnet.serialization.jsonl_stream import JSONLStream

from ..reporters.base import selector_label
from ..result import AnalysisResults
from .base import Plotter
from .registry import PlotterRegistry
from .utils import collect_scalar_values


@PlotterRegistry.register("scalar")
class ScalarPlotter(Plotter):
    """Create distribution plots for each scalar value key.

    For each (predictions_reader, selector) combination a per-key directory is
    created containing a histogram, box plot, and violin plot.

    Args:
        plotter_type: Registered plotter name from the analysis configuration.
        bins: Number of histogram bins. Defaults to ``DEFAULT_BINS``.
    """

    DEFAULT_BINS: ClassVar[int] = 50

    def __init__(self, plotter_type: str, bins: int | None = None) -> None:
        """Initialize a scalar distribution plotter."""
        super().__init__(plotter_type)
        self.bins = bins if bins is not None else self.DEFAULT_BINS

    def plot(self, results: AnalysisResults, output_dir: Path) -> None:
        """Create histogram, box, and violin PDFs for scalar values.

        Args:
            results: Analysis pipeline outputs to plot.
            output_dir: Directory where the ``scalar_plots`` tree should be written.
        """
        root = output_dir / "scalar_plots"

        for reader_idx, reader_selectors in enumerate(results.selectors):
            logging.info(f"    Predictions {reader_idx + 1}/{len(results.selectors)}.")

            for sel_idx, selector in enumerate(reader_selectors):
                logging.info(f"      Selector {sel_idx + 1}/{len(reader_selectors)}.")
                sel_label_str = selector_label(results.selectors_config, sel_idx)
                sel_cfg = results.selectors_config[sel_idx] if sel_idx < len(results.selectors_config) else {}

                stream: JSONLStream | None = None
                if reader_idx < len(results.collector_results) and sel_idx < len(results.collector_results[reader_idx]):
                    stream = results.collector_results[reader_idx][sel_idx]

                values_by_key = collect_scalar_values(selector, stream)
                if not values_by_key:
                    continue

                combo_label = f"pred_{reader_idx:03d}__sel_{sel_idx:03d}_{sel_label_str}"
                combo_dir = root / combo_label
                subtitle = _subtitle(sel_cfg, reader_idx)

                for key, vals in values_by_key.items():
                    key_dir = combo_dir / key
                    key_dir.mkdir(parents=True, exist_ok=True)
                    arr = np.array(vals)

                    self._histogram(arr, key, subtitle, key_dir)
                    self._boxplot(arr, key, subtitle, key_dir)
                    self._violin(arr, key, subtitle, key_dir)

    def _histogram(self, arr: np.ndarray, key: str, subtitle: str, out: Path) -> None:
        """Write a histogram PDF for one scalar key.

        Args:
            arr: One-dimensional scalar values with shape ``(N,)``.
            key: Scalar value key used for labels and output path context.
            subtitle: Plot subtitle text describing prediction and selector context.
            out: Directory where ``histogram.pdf`` should be written.
        """
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.hist(arr, bins=self.bins, edgecolor="black", alpha=0.75)
        ax.set_xlabel(key)
        ax.set_ylabel("Count")
        ax.set_title(f"Histogram of {key}")
        _add_subtitle(fig, subtitle)
        _add_stats_text(ax, arr)
        fig.tight_layout()
        fig.savefig(out / "histogram.pdf", bbox_inches="tight")
        plt.close(fig)

    @staticmethod
    def _boxplot(arr: np.ndarray, key: str, subtitle: str, out: Path) -> None:
        """Write a box plot PDF for one scalar key.

        Args:
            arr: One-dimensional scalar values with shape ``(N,)``.
            key: Scalar value key used for labels and output path context.
            subtitle: Plot subtitle text describing prediction and selector context.
            out: Directory where ``boxplot.pdf`` should be written.
        """
        fig, ax = plt.subplots(figsize=(5, 4.5))
        bp = ax.boxplot(arr, vert=True, patch_artist=True)
        bp["boxes"][0].set_facecolor("#005186")
        bp["boxes"][0].set_alpha(0.7)
        ax.set_ylabel(key)
        ax.set_xticklabels([""])
        ax.set_title(f"Box plot of {key}")
        _add_subtitle(fig, subtitle)
        fig.tight_layout()
        fig.savefig(out / "boxplot.pdf", bbox_inches="tight")
        plt.close(fig)

    @staticmethod
    def _violin(arr: np.ndarray, key: str, subtitle: str, out: Path) -> None:
        """Write a violin plot PDF for one scalar key.

        Args:
            arr: One-dimensional scalar values with shape ``(N,)``.
            key: Scalar value key used for labels and output path context.
            subtitle: Plot subtitle text describing prediction and selector context.
            out: Directory where ``violin.pdf`` should be written.
        """
        fig, ax = plt.subplots(figsize=(5, 4.5))
        vp = ax.violinplot(arr, showmedians=True, showextrema=True)
        bodies = vp["bodies"]
        assert isinstance(bodies, list)
        for body in bodies:
            body.set_facecolor("#005186")
            body.set_alpha(0.7)
        ax.set_ylabel(key)
        ax.set_xticks([1])
        ax.set_xticklabels([""])
        ax.set_title(f"Violin plot of {key}")
        _add_subtitle(fig, subtitle)
        fig.tight_layout()
        fig.savefig(out / "violin.pdf", bbox_inches="tight")
        plt.close(fig)


def _subtitle(sel_cfg: dict[str, Any], reader_idx: int) -> str:
    """Build the scalar plot subtitle for one prediction-reader/selector pair.

    Args:
        sel_cfg: Selector configuration dictionary for this selector index.
        reader_idx: Zero-based prediction reader index.

    Returns:
        Human-readable subtitle string.
    """
    parts = [f"predictions={reader_idx}"]
    sel_type = sel_cfg.get("selector_type", "?")
    parts.append(f"selector={sel_type}")
    extras = {k: v for k, v in sel_cfg.items() if k != "selector_type"}
    if extras:
        parts.append(" ".join(f"{k}={v}" for k, v in extras.items()))
    return "  |  ".join(parts)


def _add_subtitle(fig: Figure, text: str) -> None:
    """Add centered subtitle text below a figure.

    Args:
        fig: Matplotlib figure to annotate.
        text: Subtitle text.
    """
    fig.text(0.5, -0.01, text, ha="center", va="top", fontsize=7, color="gray")


def _add_stats_text(ax: Axes, arr: np.ndarray) -> None:
    """Add sample count and summary statistics to a plot axis.

    Args:
        ax: Matplotlib axis to annotate.
        arr: One-dimensional scalar values with shape ``(N,)``.
    """
    text = f"n={len(arr)}\n" f"mean={np.mean(arr):.4g}\n" f"std={np.std(arr):.4g}\n" f"median={np.median(arr):.4g}"
    ax.text(
        0.97,
        0.95,
        text,
        transform=ax.transAxes,
        fontsize=7,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.5),
    )
