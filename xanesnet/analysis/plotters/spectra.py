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
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

from .base import Plotter
from .registry import PlotterRegistry

# Use non-interactive backend
matplotlib.use("Agg")


@PlotterRegistry.register("spectra")
class SpectraPlotter(Plotter):
    """
    Plot predicted vs target spectra for selected samples.

    All plots are saved in a single PDF file.
    """

    def __init__(
        self,
        plotter_type: str,
        pred_key: str = "prediction",
        target_key: str = "target",
        value_keys: list[str] | None = None,
        sort_by_value: bool = False,
        samples_per_page: int = 3,
        figsize: tuple[int, int] = (10, 10),
    ) -> None:
        super().__init__(plotter_type)
        self.pred_key = pred_key
        self.target_key = target_key
        self.value_keys = value_keys if value_keys is not None else []
        self.sort_by_value = sort_by_value
        self.samples_per_page = samples_per_page
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
        # Get sample data from selector
        sample_data = list(selector)

        if not sample_data:
            logging.warning("No samples found for spectra plotting")
            return

        num_samples = len(sample_data)

        # Create PDF
        output_path = output_dir / "spectra_comparison.pdf"

        # Scientific styling
        plt.rcParams.update(
            {
                "font.size": 10,
                "axes.labelsize": 11,
                "axes.titlesize": 11,
                "xtick.labelsize": 9,
                "ytick.labelsize": 9,
                "legend.fontsize": 9,
                "font.family": "serif",
                "axes.linewidth": 1.0,
                "grid.linewidth": 0.5,
                "grid.alpha": 0.3,
            }
        )

        with PdfPages(output_path) as pdf:
            # First page: Configuration
            self._plot_config_page(selector_config, per_sample_configs, aggregator_configs, pdf)

            # Second page: Distribution plot
            self._plot_distribution(sample_data, pdf)

            # Determine sample order (with or without sorting)
            sample_indices = list(range(num_samples))
            if self.sort_by_value and self.value_keys:
                # Sort by first value_key
                first_value_key = self.value_keys[0]
                # Create list of (index, value) tuples
                index_value_pairs = []
                for idx in range(num_samples):
                    value = per_sample_results[idx].get(first_value_key)
                    if value is not None:
                        index_value_pairs.append((idx, value))
                    else:
                        # Keep samples without values at end
                        index_value_pairs.append((idx, float("inf")))
                # Sort by value (ascending)
                index_value_pairs.sort(key=lambda x: x[1])
                sample_indices = [idx for idx, _ in index_value_pairs]
                logging.info(f"Sorted {num_samples} samples by '{first_value_key}'")

            # Plot samples in batches (pages)
            for page_start in range(0, num_samples, self.samples_per_page):
                page_end = min(page_start + self.samples_per_page, num_samples)
                page_samples = page_end - page_start

                # Always create subplots for full samples_per_page to maintain consistent sizing
                fig, axes = plt.subplots(self.samples_per_page, 1, figsize=self.figsize, squeeze=False)
                axes = axes.flatten()

                for idx in range(page_samples):
                    sample_idx = sample_indices[page_start + idx]
                    sample = sample_data[sample_idx]
                    sample_results = per_sample_results[sample_idx]
                    ax = axes[idx]

                    pred = sample.get(self.pred_key)
                    target = sample.get(self.target_key)

                    if pred is None or target is None:
                        logging.warning(
                            f"Sample {sample_idx} missing '{self.pred_key}' or '{self.target_key}', skipping"
                        )
                        ax.text(
                            0.5,
                            0.5,
                            f"Sample {sample_idx}\nMissing data",
                            ha="center",
                            va="center",
                            transform=ax.transAxes,
                            fontsize=10,
                        )
                        ax.set_xticks([])
                        ax.set_yticks([])
                        continue

                    # Convert to numpy
                    if not isinstance(pred, np.ndarray):
                        pred = np.array(pred)
                    if not isinstance(target, np.ndarray):
                        target = np.array(target)

                    # Plot with scientific styling
                    x = np.arange(len(pred))
                    ax.plot(x, target, label="Ground Truth", linewidth=1.5, alpha=0.9, color="#000000", zorder=2)
                    ax.plot(x, pred, label="Prediction", linewidth=1.2, alpha=0.8, color="#DC143C", zorder=1)

                    # Get sample_id if available
                    sample_id = sample.get("sample_id", sample_idx)
                    ax.set_title(f"Sample {sample_id}", fontweight="bold", fontsize=11, pad=8)
                    ax.set_ylabel("Intensity", fontweight="bold")
                    ax.legend(loc="upper right", fontsize=9, framealpha=0.9, edgecolor="gray")
                    ax.grid(True, alpha=0.3, linestyle="--")
                    ax.spines["top"].set_visible(False)
                    ax.spines["right"].set_visible(False)

                    # Get value_keys from per_sample_results
                    value_text_lines = []
                    for value_key in self.value_keys:
                        value = sample_results.get(value_key)
                        if value is not None:
                            value_text_lines.append(f"{value_key} $= {value:.4f}$")

                    if value_text_lines:
                        value_text = "\n".join(value_text_lines)
                        ax.text(
                            0.02,
                            0.98,
                            value_text,
                            transform=ax.transAxes,
                            verticalalignment="top",
                            horizontalalignment="left",
                            bbox=dict(
                                boxstyle="round,pad=0.4", facecolor="white", edgecolor="gray", alpha=0.9, linewidth=1
                            ),
                            fontsize=9,
                            family="serif",
                        )

                # Hide unused axes on last page
                for idx in range(page_samples, self.samples_per_page):
                    axes[idx].set_visible(False)

                # Set xlabel only on bottom plot (last visible one)
                if page_samples > 0:
                    axes[page_samples - 1].set_xlabel("Energy Point Index", fontweight="bold")

                plt.tight_layout()
                pdf.savefig(fig, dpi=300, bbox_inches="tight")
                plt.close(fig)

        logging.info(f"Saved spectra comparison plot to {output_path} ({num_samples} samples)")

    def _plot_config_page(
        self,
        selector_config: dict[str, Any],
        per_sample_configs: list[dict[str, Any]],
        aggregator_configs: list[dict[str, Any]],
        pdf: PdfPages,
    ) -> None:
        """
        Plot configuration information as first page of PDF.
        """
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111)
        ax.axis("off")

        # Build config text
        config_lines = []
        config_lines.append("SPECTRA PLOTTER CONFIGURATION")
        config_lines.append("=" * 50)
        config_lines.append("")
        config_lines.append("Plotter Settings:")
        config_lines.append(f"  - Plotter Type: {self.plotter_type}")
        config_lines.append(f"  - Prediction Key: {self.pred_key}")
        config_lines.append(f"  - Target Key: {self.target_key}")
        config_lines.append(f"  - Value Keys: {', '.join(self.value_keys) if self.value_keys else 'None'}")
        config_lines.append(f"  - Sort by Value: {self.sort_by_value}")
        config_lines.append(f"  - Samples per Page: {self.samples_per_page}")
        config_lines.append(f"  - Figure Size: {self.figsize}")
        config_lines.append("")
        config_lines.append("Selector Configuration:")
        for key, value in selector_config.items():
            config_lines.append(f"  - {key}: {value}")
        config_lines.append("")
        config_lines.append("Per-Sample Modules:")
        for i, ps_config in enumerate(per_sample_configs):
            config_lines.append(f"  [{i+1}] {ps_config.get('per_sample_type', 'Unknown')}")
            for key, value in ps_config.items():
                if key != "per_sample_type":
                    config_lines.append(f"      - {key}: {value}")
        config_lines.append("")
        config_lines.append("Aggregators:")
        for i, agg_config in enumerate(aggregator_configs):
            config_lines.append(f"  [{i+1}] {agg_config.get('aggregator_type', 'Unknown')}")
            for key, value in agg_config.items():
                if key != "aggregator_type":
                    config_lines.append(f"      - {key}: {value}")

        # Display text
        config_text = "\n".join(config_lines)
        ax.text(
            0.05,
            0.95,
            config_text,
            transform=ax.transAxes,
            verticalalignment="top",
            horizontalalignment="left",
            fontsize=9,
            family="monospace",
            wrap=True,
        )

        pdf.savefig(fig, dpi=300, bbox_inches="tight")
        plt.close(fig)

    def _plot_distribution(self, sample_data: list[dict[str, Any]], pdf: PdfPages) -> None:
        """
        Plot the distribution of prediction and ground truth spectra.

        Shows mean and gradient representing deviation from mean.
        """
        # Collect all predictions and targets
        predictions = []
        targets = []

        for sample in sample_data:
            pred = sample.get(self.pred_key)
            target = sample.get(self.target_key)

            if pred is not None and target is not None:
                if not isinstance(pred, np.ndarray):
                    pred = np.array(pred)
                if not isinstance(target, np.ndarray):
                    target = np.array(target)
                predictions.append(pred)
                targets.append(target)

        if not predictions:
            logging.warning("No valid predictions/targets for distribution plot")
            return

        # Convert to arrays
        predictions = np.array(predictions)  # Shape: (n_samples, n_points)
        targets = np.array(targets)

        # Compute statistics
        pred_mean = np.mean(predictions, axis=0)
        pred_std = np.std(predictions, axis=0)
        target_mean = np.mean(targets, axis=0)
        target_std = np.std(targets, axis=0)

        x = np.arange(len(pred_mean))

        # Create plot
        fig, ax = plt.subplots(figsize=(10, 6))

        n_samples = len(targets)

        # Simple overlay with broader lines and low transparency
        # Natural intensity gradient: dense where many overlap (near mean), light at extremes
        alpha = 0.15  # Low transparency for overlay effect
        linewidth = 1.5  # Broader lines for better visibility

        # Plot all ground truth spectra
        for target_spectrum in targets:
            ax.plot(x, target_spectrum, linewidth=linewidth, alpha=alpha, color="#000000", zorder=1)

        # Plot all prediction spectra
        for pred_spectrum in predictions:
            ax.plot(x, pred_spectrum, linewidth=linewidth, alpha=alpha, color="#DC143C", zorder=2)

        # Plot means on top for reference
        ax.plot(x, target_mean, label="Ground Truth (mean)", linewidth=2.5, alpha=0.95, color="#000000", zorder=4)
        ax.plot(
            x,
            pred_mean,
            label="Prediction (mean)",
            linewidth=2.5,
            alpha=0.95,
            color="#DC143C",
            zorder=3,
        )

        # Add legend explanation
        ax.plot(
            [],
            [],
            color="#666666",
            linewidth=linewidth,
            alpha=alpha * 2,
            label=f"Individual spectra (n={n_samples}, α={alpha})",
        )

        ax.set_xlabel("Energy Point Index", fontweight="bold")
        ax.set_ylabel("Intensity", fontweight="bold")
        ax.set_title(
            f"Spectra Distribution with Deviation Gradient (n={len(predictions)} samples)",
            fontweight="bold",
            pad=15,
        )
        ax.legend(loc="best", fontsize=9, framealpha=0.9, edgecolor="gray")
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        plt.tight_layout()
        pdf.savefig(fig, dpi=300, bbox_inches="tight")
        plt.close(fig)
