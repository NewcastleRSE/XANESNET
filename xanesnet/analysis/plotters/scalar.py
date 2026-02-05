"""
XANESNET

This program is free software: you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free Software
Foundation, either Version 3 of the License, or (at your option) any later
version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with
this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import logging
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Literal

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

from .base import Plotter
from .registry import PlotterRegistry

# Use non-interactive backend
matplotlib.use("Agg")


@PlotterRegistry.register("scalar")
class ScalarPlotter(Plotter):
    """
    Plot scalar values from per-sample results.

    Supports different plot types: histogram, boxplot, violin
    """

    def __init__(
        self,
        plotter_type: str,
        value_key: str,
        plot_type: Literal["histogram", "boxplot", "violin"] = "histogram",
        bins: int = 50,
        figsize: tuple[int, int] = (10, 6),
    ) -> None:
        super().__init__(plotter_type)
        self.value_key = value_key
        self.plot_type = plot_type
        self.bins = bins
        self.figsize = figsize

    def plot(
        self,
        selector: Iterable[dict[str, Any]],
        selector_config: dict[str, Any],
        per_sample_results: list[dict[str, Any]],
        per_sample_configs: list[dict[str, Any]],
        aggregated_results: list[dict[str, Any]],
        aggregator_configs: list[dict[str, Any]],
        output_dir: Path,
    ) -> None:
        if not per_sample_results:
            logging.warning("No per-sample results available for scalar plot")
            return

        # Extract values for the specified key
        values = [sample.get(self.value_key) for sample in per_sample_results if self.value_key in sample]

        if not values:
            logging.warning(f"Value key '{self.value_key}' not found in per-sample results")
            return

        values_arr = np.array(values)

        # Create output path
        output_path = output_dir / f"scalar_{self.value_key}_{self.plot_type}.pdf"

        with PdfPages(output_path) as pdf:
            # Create plot based on type
            fig, ax = plt.subplots(figsize=self.figsize)

            # Scientific styling
            plt.rcParams.update(
                {
                    "font.size": 11,
                    "axes.labelsize": 12,
                    "axes.titlesize": 13,
                    "xtick.labelsize": 10,
                    "ytick.labelsize": 10,
                    "legend.fontsize": 10,
                    "font.family": "serif",
                    "axes.linewidth": 1.2,
                    "grid.linewidth": 0.8,
                    "grid.alpha": 0.3,
                }
            )

            if self.plot_type == "histogram":
                counts, bins, patches = ax.hist(
                    values_arr, bins=self.bins, edgecolor="black", linewidth=0.5, alpha=0.75, color="#4682B4"
                )
                ax.set_ylabel("Frequency", fontweight="bold")
                ax.set_xlabel(self.value_key, fontweight="bold")
            elif self.plot_type == "boxplot":
                bp = ax.boxplot(
                    values_arr,
                    vert=True,
                    patch_artist=True,
                    boxprops=dict(facecolor="#4682B4", alpha=0.7),
                    medianprops=dict(color="red", linewidth=2),
                    whiskerprops=dict(linewidth=1.2),
                    capprops=dict(linewidth=1.2),
                )
                ax.set_ylabel(self.value_key, fontweight="bold")
                ax.set_xticklabels([self.value_key])
            elif self.plot_type == "violin":
                parts = ax.violinplot([values_arr], vert=True, showmeans=True, showmedians=True, showextrema=True)
                for pc in parts["bodies"]:
                    pc.set_facecolor("#4682B4")
                    pc.set_alpha(0.7)
                ax.set_ylabel(self.value_key, fontweight="bold")
                ax.set_xticks([1])
                ax.set_xticklabels([self.value_key])

            ax.set_title(f"Distribution of {self.value_key}", fontweight="bold", pad=15)
            ax.grid(True, alpha=0.3, axis="y", linestyle="--")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            # Add statistics text box
            stats_text = (
                f"$n = {len(values_arr)}$\n"
                f"$\\mu = {np.mean(values_arr):.4f}$\n"
                f"$\\sigma = {np.std(values_arr):.4f}$\n"
                f"median $= {np.median(values_arr):.4f}$\n"
                f"min $= {np.min(values_arr):.4f}$\n"
                f"max $= {np.max(values_arr):.4f}$"
            )
            ax.text(
                0.98,
                0.98,
                stats_text,
                transform=ax.transAxes,
                verticalalignment="top",
                horizontalalignment="right",
                bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="gray", alpha=0.9, linewidth=1),
                fontsize=10,
                family="serif",
            )

            plt.tight_layout()
            pdf.savefig(fig, dpi=300, bbox_inches="tight")
            plt.close(fig)

        logging.info(f"Saved scalar plot to {output_path}")
