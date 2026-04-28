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

"""Plotter that renders aggregated statistics as table PDFs."""

import logging
from pathlib import Path
from typing import Any, ClassVar, cast

import matplotlib.pyplot as plt

from ..reporters.base import selector_label
from ..result import AnalysisResults
from .base import Plotter
from .registry import PlotterRegistry


@PlotterRegistry.register("stat_table")
class StatTablePlotter(Plotter):
    """Render comparison tables of aggregated statistics as PDF figures.

    For each scalar value key found across aggregator results, a table is
    produced where rows are prediction-reader/selector combinations and columns are statistics such
    as ``mean``, ``std``, and ``median``.

    Args:
        plotter_type: Registered plotter name from the analysis configuration.
        stat_keys: Ordered statistic keys to include as table columns.
        precision: Number of significant digits used when formatting table values.
    """

    DEFAULT_STAT_KEYS: ClassVar[list[str]] = ["mean", "std", "median", "min", "max"]

    def __init__(
        self,
        plotter_type: str,
        stat_keys: list[str] | None = None,
        precision: int = 4,
    ) -> None:
        """Initialize a statistics table plotter."""
        super().__init__(plotter_type)
        self.stat_keys = stat_keys if stat_keys is not None else self.DEFAULT_STAT_KEYS
        self.precision = precision

    def plot(self, results: AnalysisResults, output_dir: Path) -> None:
        """Write table PDFs for aggregated scalar statistics.

        Args:
            results: Analysis pipeline outputs to plot.
            output_dir: Directory where the ``stat_tables`` tree should be written.
        """
        if not results.aggregator_results:
            logging.info("    No aggregator results available, skipping.")
            return

        root = output_dir / "stat_tables"

        table_data: dict[tuple[str, int, str], dict[str, dict[str, float]]] = {}

        for reader_idx, reader_results in enumerate(results.aggregator_results):
            logging.info(f"    Predictions {reader_idx + 1}/{len(results.aggregator_results)}.")

            for sel_idx, agg_results in enumerate(reader_results):
                sel_label_str = selector_label(results.selectors_config, sel_idx)
                sel_cfg = results.selectors_config[sel_idx] if sel_idx < len(results.selectors_config) else {}
                row_label = _row_label(reader_idx, sel_idx, sel_label_str, sel_cfg)

                for agg_result in agg_results:
                    for value_key, stats in agg_result.data.items():
                        if not isinstance(stats, dict):
                            continue
                        table_key = (agg_result.aggregator_type, agg_result.aggregator_index, value_key)
                        table_data.setdefault(table_key, {})[row_label] = cast(dict[str, float], stats)

        if not table_data:
            logging.info("    No table data collected, skipping.")
            return

        for (agg_type, agg_idx, value_key), rows in table_data.items():
            agg_dir = root / f"{agg_type}_{agg_idx:03d}"
            agg_dir.mkdir(parents=True, exist_ok=True)
            filepath = agg_dir / f"{value_key}.pdf"

            self._render_table(rows, value_key, agg_type, agg_idx, filepath)

    def _render_table(
        self,
        rows: dict[str, dict[str, float]],
        value_key: str,
        agg_type: str,
        agg_idx: int,
        filepath: Path,
    ) -> None:
        """Render a single comparison table to a PDF using Matplotlib.

        Args:
            rows: Mapping from row label to statistic values.
            value_key: Scalar value key represented by the table.
            agg_type: Registered aggregator name that produced the statistics.
            agg_idx: Zero-based aggregator index from the analysis configuration.
            filepath: Destination PDF path.
        """
        row_labels = list(rows.keys())
        col_labels = [s for s in self.stat_keys if any(s in stats for stats in rows.values())]

        if not col_labels or not row_labels:
            return

        cell_text: list[list[str]] = []
        cell_values: list[list[float | None]] = []
        for rl in row_labels:
            stats = rows[rl]
            text_row: list[str] = []
            val_row: list[float | None] = []
            for cl in col_labels:
                v = stats.get(cl)
                if v is not None:
                    text_row.append(f"{v:.{self.precision}g}")
                    val_row.append(v)
                else:
                    text_row.append("-")
                    val_row.append(None)
            cell_text.append(text_row)
            cell_values.append(val_row)

        cell_colours = self._cell_colours(cell_values)

        n_rows, n_cols = len(row_labels), len(col_labels)
        fig_width = max(6, 1.8 * n_cols + 3)
        fig_height = max(2, 0.45 * n_rows + 1.6)

        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        ax.axis("off")

        table = ax.table(
            cellText=cell_text,
            rowLabels=row_labels,
            colLabels=col_labels,
            cellColours=cell_colours,
            loc="center",
            cellLoc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1, 1.4)

        for (r, c), cell in table.get_celld().items():
            if r == 0:
                cell.set_facecolor("#204aff")
                cell.set_text_props(color="white", weight="bold")
            if c == -1:
                cell.set_text_props(fontsize=7, ha="right")

        ax.set_title(
            f"Statistics for '{value_key}'  (aggregator: {agg_type} #{agg_idx})",
            fontsize=10,
            pad=12,
        )

        fig.tight_layout()
        fig.savefig(filepath, bbox_inches="tight")
        plt.close(fig)

    @staticmethod
    def _cell_colours(cell_values: list[list[float | None]]) -> list[list[str]]:
        """Generate per-cell background colours.

        Best (lowest) value in each column gets a green tint;
        worst (highest) gets a light red. Others stay white.

        Args:
            cell_values: Numeric table values with missing values represented by ``None``.

        Returns:
            Matrix of Matplotlib-compatible colour strings matching ``cell_values``.
        """
        n_rows = len(cell_values)
        n_cols = len(cell_values[0]) if cell_values else 0
        colours: list[list[str]] = [["white"] * n_cols for _ in range(n_rows)]

        if n_rows < 2:
            return colours

        for c in range(n_cols):
            col_vals: list[tuple[int, float]] = []
            for r in range(n_rows):
                v = cell_values[r][c]
                if v is not None:
                    col_vals.append((r, v))
            if len(col_vals) < 2:
                continue
            sorted_vals = sorted(col_vals, key=lambda x: x[1])
            best_row = sorted_vals[0][0]
            worst_row = sorted_vals[-1][0]
            colours[best_row][c] = "#d5f5d5"
            colours[worst_row][c] = "#f5d5d5"

        return colours


def _row_label(reader_idx: int, sel_idx: int, sel_label_str: str, sel_cfg: dict[str, Any]) -> str:
    """Build a descriptive row label for a statistics table.

    Args:
        reader_idx: Zero-based prediction reader index.
        sel_idx: Zero-based selector index.
        sel_label_str: Selector label derived from configuration.
        sel_cfg: Selector configuration dictionary for this selector index.

    Returns:
        Human-readable row label.
    """
    parts = [f"pred={reader_idx}", f"sel={sel_label_str}"]
    extras = {k: v for k, v in sel_cfg.items() if k != "selector_type"}
    if extras:
        parts.append(" ".join(f"{k}={v}" for k, v in extras.items()))
    return "  |  ".join(parts)
