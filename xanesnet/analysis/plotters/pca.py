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

"""Plotter for structure-group separation in PCA space."""

import logging
from pathlib import Path
from typing import Any, cast

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes

from xanesnet.serialization.config import Config
from xanesnet.utils.exceptions import ConfigError

from ..result import AnalysisResults
from ..selectors import Selector
from ..utils import as_float_vector
from .base import Plotter
from .common.layout import finish_single_panel, save_figure
from .common.style import PlotSize, add_legend, method_color, set_context_title
from .registry import PlotterRegistry

_PCA_3D_OUTSIDE_LEGEND_ANCHOR = (1.20, 1.0)


@PlotterRegistry.register("pca")
class PcaPlotter(Plotter):
    """Plot a reader's structure groups in PCA space.

    Descriptor vectors come from a configured collector and selectors are
    assumed disjoint.

    Requires:
        Descriptor vectors: provided by a collector emitting ``descriptor_key``.

    Args:
        plotter_type: Registered plotter name from the analysis configuration.
        latex_font: Render figures in a LaTeX-style serif font when ``True``.
        descriptor_key: Collector key holding the descriptor vectors.
        legend_position: Place the selector legend ``"inside"`` the axes or
            ``"outside"`` it on the right.
        plot_size: Shared figure size profile: ``"small"`` or ``"default"``.
    """

    def __init__(
        self,
        plotter_type: str,
        descriptor_key: str,
        legend_position: str,
        latex_font: bool,
        plot_size: PlotSize,
    ) -> None:
        """Initialize a PCA plotter."""
        super().__init__(plotter_type, latex_font=latex_font, plot_size=plot_size)
        self.descriptor_key = descriptor_key
        self.legend_position = legend_position

    def _plot(self, results: AnalysisResults, output_dir: Path) -> None:
        """Write one 2D and one 3D PCA scatter per prediction reader.

        Args:
            results: Analysis pipeline outputs to plot.
            output_dir: Directory where the ``pca_plots`` tree should be written.

        Raises:
            ConfigError: If ``descriptor_key`` is missing from a collector record.
        """
        if not results.selectors:
            logging.info("    No selectors available, skipping.")
            return

        root = output_dir / "pca_plots"
        root.mkdir(parents=True, exist_ok=True)

        for reader_idx, reader_selectors in enumerate(results.selectors):
            logging.info(f"    Predictions {reader_idx + 1}/{len(results.selectors)}.")

            projections, groups = self._collect_projections(results, reader_idx)
            if projections is None:
                continue

            reader_dir = root / f"pred_{reader_idx:03d}"
            reader_dir.mkdir(parents=True, exist_ok=True)
            subtitle = results.prediction_names[reader_idx]

            self._scatter(projections, groups, reader_selectors, subtitle, (0, 1), reader_dir / "pca_2d.pdf")
            if projections.shape[1] >= 3:
                self._scatter(
                    projections,
                    groups,
                    reader_selectors,
                    subtitle,
                    (0, 1, 2),
                    reader_dir / "pca_3d.pdf",
                )

    def _collect_projections(self, results: AnalysisResults, reader_idx: int) -> tuple[np.ndarray | None, list[int]]:
        """Pool the descriptor vectors of every selector and project them.

        Args:
            results: Analysis pipeline outputs to plot.
            reader_idx: Zero-based prediction reader index.

        Returns:
            ``(projections, groups)`` where ``projections`` has shape
            ``(M, n_components)`` and ``groups`` holds the selector index of
            each row, or ``(None, [])`` when there are too few vectors.

        Raises:
            ConfigError: If ``descriptor_key`` is missing from a collector record.
        """
        vectors: list[np.ndarray] = []
        groups: list[int] = []
        reader_selectors = results.selectors[reader_idx]

        for sel_idx in range(len(reader_selectors)):
            stream = results.collector_stream(reader_idx, sel_idx)
            if stream is None:
                continue
            for record in stream:
                value = record.get(self.descriptor_key)
                if value is None:
                    raise ConfigError(
                        f"Key '{self.descriptor_key}' is missing from a collector record. "
                        "Configure a descriptor collector that produces this key."
                    )
                vectors.append(as_float_vector(value))
                groups.append(sel_idx)

        if len(vectors) < 2:
            logging.info("      Need at least two descriptor vectors for PCA, skipping.")
            return None, []

        projections = self._project(np.stack(vectors))
        if projections.shape[1] < 2:
            logging.info("      Descriptor is too low-dimensional for PCA, skipping.")
            return None, []
        return projections, groups

    @staticmethod
    def _project(matrix: np.ndarray) -> np.ndarray:
        """Standardize and project a feature matrix onto its leading components.

        Args:
            matrix: Feature matrix with shape ``(M, D)``.

        Returns:
            Projection with shape ``(M, n_components)`` where
            ``n_components = min(3, M, D)``.
        """
        with np.errstate(invalid="ignore", divide="ignore"):
            scaled = (matrix - matrix.mean(axis=0)) / matrix.std(axis=0)
        scaled = np.nan_to_num(scaled, nan=0.0, posinf=0.0, neginf=0.0)
        centered = scaled - scaled.mean(axis=0)
        n_components = min(3, centered.shape[0], centered.shape[1])
        if centered.shape[1] < 2:
            return centered[:, :1]
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        return centered @ vt[:n_components].T

    def _draw_group_scatter(
        self,
        ax: Axes,
        projections: np.ndarray,
        groups: list[int],
        selectors: list[Selector],
        dims: tuple[int, ...],
    ) -> None:
        """Draw one per-selector scatter of the projected points.

        Args:
            ax: Matplotlib axis to draw on.
            projections: Projection matrix with shape ``(M, n_components)``.
            groups: Selector index of each projected row.
            selectors: Selector instances in selector order.
            dims: Indices of the components to draw.
        """
        group_ids = np.asarray(groups, dtype=int)
        for sel_idx in sorted(set(groups)):
            mask = group_ids == sel_idx
            coords = [projections[mask, dim] for dim in dims]
            kwargs: dict[str, Any] = {
                "color": method_color(sel_idx),
                "s": self.style.scatter_size("pca"),
                "alpha": 0.85,
            }
            if len(dims) == 3:
                kwargs["depthshade"] = False
            ax.scatter(*coords, label=str(selectors[sel_idx]), **kwargs)

    def _scatter(
        self,
        projections: np.ndarray,
        groups: list[int],
        selectors: list[Selector],
        subtitle: str,
        dims: tuple[int, ...],
        out: Path,
    ) -> None:
        """Write one PCA scatter figure for a projection.

        Args:
            projections: Projection matrix with shape ``(M, n_components)``.
            groups: Selector index of each projected row.
            selectors: Selector instances in selector order.
            subtitle: Plot subtitle text describing the prediction reader.
            dims: Indices of the components to draw.
            out: Destination PDF path.
        """
        figure_size = self.style.figure_size(
            "square",
            width_margin=self.style.legend_margin(self.legend_position),
        )
        fig = plt.figure(figsize=figure_size)

        if len(dims) == 3:
            from mpl_toolkits import mplot3d

            _ = mplot3d.Axes3D  # Register the 3D projection.

            ax = cast(Any, fig.add_subplot(111, projection="3d"))
            finish_single_panel(fig, ax, self.style, "square", box_aspect=(1, 1, 1))
            ax.set_zlabel("PC 3")
        else:
            ax = fig.add_subplot(111)
            finish_single_panel(fig, ax, self.style, "square")

        self._draw_group_scatter(ax, projections, groups, selectors, dims)
        ax.set_xlabel("PC 1")
        ax.set_ylabel("PC 2")
        legend_kwargs: dict[str, Any] = {}
        if len(dims) == 3 and self.legend_position == "outside":
            # 3D panels use part of the outside band for the PC3 axis label.
            legend_kwargs["bbox_to_anchor"] = _PCA_3D_OUTSIDE_LEGEND_ANCHOR
        add_legend(ax, self.style, position=self.legend_position, **legend_kwargs)
        set_context_title(ax, subtitle, self.style)
        save_figure(fig, out, self.style, bbox_inches="tight")

    @property
    def signature(self) -> Config:
        """Return the PCA plotter signature.

        Returns:
            Configuration values needed to recreate this plotter.
        """
        signature = super().signature
        signature.update_with_dict({"descriptor_key": self.descriptor_key, "legend_position": self.legend_position})
        return signature
