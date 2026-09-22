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

"""Plotter that renders aggregated statistics as table PDFs."""

import logging
from pathlib import Path

import matplotlib.pyplot as plt

from xanesnet.serialization.config import Config

from ..result import AnalysisResults
from ..utils import one_line_label
from .base import Plotter
from .common import stat_tables
from .common.formatting import truncate_text
from .common.layout import save_figure
from .common.style import PlotSize
from .registry import PlotterRegistry

_MARK_COLORS: dict[str, str] = {"best": "#d5f5d5", "worst": "#f5d5d5"}

# Marker appended to the header of the column whose values order the rows.
_SORT_INDICATOR = " \u2193"


@PlotterRegistry.register("stat_table")
class StatTablePlotter(Plotter):
    """Render comparison tables of aggregated statistics as PDF figures.

    Rows are prediction-reader/selector combinations and columns are the
    configured statistics. A compact combined table is rendered per aggregator.

    Requires:
        Scalar statistics: provided by at least one aggregator.

    Args:
        plotter_type: Registered plotter name from the analysis configuration.
        latex_font: Render figures in a LaTeX-style serif font when ``True``.
        stat_keys: Ordered statistic keys used as table columns and as
            sub-rows of the combined table.
        precision: Number of significant digits used when formatting table values.
        sort_key: Scalar value key whose first statistic orders the combined
            table rows. ``None`` uses the first value key.
        plot_size: Shared figure size profile: ``"small"`` or ``"default"``.
    """

    def __init__(
        self,
        plotter_type: str,
        stat_keys: list[str],
        precision: int,
        sort_key: str | None,
        latex_font: bool,
        plot_size: PlotSize,
    ) -> None:
        """Initialize a statistics table plotter."""
        super().__init__(plotter_type, latex_font=latex_font, plot_size=plot_size)
        self.stat_keys = stat_keys
        self.precision = precision
        self.sort_key = sort_key

    def _plot(self, results: AnalysisResults, output_dir: Path) -> None:
        """Write table PDFs for aggregated scalar statistics.

        Args:
            results: Analysis pipeline outputs to plot.
            output_dir: Directory where the ``stat_tables`` tree should be written.
        """
        if not results.aggregator_results:
            logging.info("    No aggregator results available, skipping.")
            return

        root = output_dir / "stat_tables"
        source = stat_tables.collect_tables(results)

        if not source.single:
            logging.info("    No table data collected, skipping.")
            return

        for (agg_type, agg_idx, value_key), rows in source.single.items():
            table = stat_tables.build_single_table(rows, self.stat_keys, self.precision)
            if table is None:
                continue
            agg_dir = root / stat_tables.aggregator_dir_name(agg_type, agg_idx)
            agg_dir.mkdir(parents=True, exist_ok=True)
            self._render_table(table, agg_dir / f"{value_key}.pdf")

        for (agg_type, agg_idx), rows in source.combined.items():
            value_keys = source.value_order.get((agg_type, agg_idx), [])
            table = stat_tables.build_combined_table(
                rows, source.row_order, value_keys, self.stat_keys, self.precision, self.sort_key
            )
            if table is None:
                continue
            agg_dir = root / stat_tables.aggregator_dir_name(agg_type, agg_idx)
            agg_dir.mkdir(parents=True, exist_ok=True)
            stem = stat_tables.combined_stem(value_keys)
            self._render_combined_table(table, source.row_label_lines, agg_dir / f"{stem}.pdf")

        logging.info(f"    Wrote table PDFs to '{root}'.")

    def _render_table(self, table: stat_tables.SingleTable, filepath: Path) -> None:
        """Render a single comparison table to a PDF using Matplotlib.

        Args:
            table: Laid-out single-value table.
            filepath: Destination PDF path.
        """
        row_labels = table.row_labels
        col_labels = _annotate_sort_column(table.col_labels, table.sort_col, self.style.table_label_width())
        cell_colors = _cell_colors(table.cell_marks)

        n_rows, n_cols = len(row_labels), len(col_labels)
        fig_width, fig_height = self.style.figsize((max(4, 1.4 * n_cols + 2), max(1.6, 0.38 * n_rows + 1.2)))

        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        ax.axis("off")

        mpl_table = ax.table(
            cellText=table.cell_text,
            rowLabels=row_labels,
            colLabels=col_labels,
            cellColours=cell_colors,
            loc="center",
            cellLoc="center",
        )
        mpl_table.auto_set_font_size(False)
        mpl_table.set_fontsize(self.style.fontsize("table"))
        mpl_table.scale(1, 1.2 * self.style.layout_scale)

        for (r, c), cell in mpl_table.get_celld().items():
            cell.PAD = self.style.spacing("table_pad")
            if r == 0:
                cell.set_facecolor("#f0f0f0")
                cell.set_text_props(weight="bold")
            if c == -1:
                cell.set_text_props(fontsize=self.style.fontsize("table"), ha="right")

        fig.tight_layout(pad=self.style.table_tight_layout_pad)
        save_figure(fig, filepath, self.style)

    def _render_combined_table(
        self,
        table: stat_tables.CombinedTable,
        row_label_lines: dict[str, list[str]],
        filepath: Path,
    ) -> None:
        """Render one compact combined table for all scalar value keys to a PDF.

        Args:
            table: Laid-out combined table.
            row_label_lines: Label lines per method row label used for the
                merged group cells.
            filepath: Destination PDF path.
        """
        value_cols = _annotate_sort_column(
            [stat_tables.value_display_label(value_key) for value_key in table.value_cols],
            table.sort_col,
            self.style.table_label_width(),
        )
        cell_colors = [["white"] + colors_row for colors_row in _cell_colors(table.cell_marks)]

        n_rows = len(table.cell_text)
        n_cols = 1 + len(value_cols)
        fontsize = self.style.fontsize("table")
        fig_width, fig_height = self.style.figsize((max(4, 1.1 * n_cols + 2), max(2.0, 0.3 * n_rows + 1.4)))

        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        ax.axis("off")

        mpl_table = ax.table(
            cellText=table.cell_text,
            rowLabels=[""] * n_rows,
            colLabels=[""] + value_cols,
            cellColours=cell_colors,
            loc="center",
            cellLoc="center",
        )
        mpl_table.auto_set_font_size(False)
        mpl_table.set_fontsize(fontsize)
        mpl_table.scale(1, 1.15 * self.style.layout_scale)

        for (r, c), cell in mpl_table.get_celld().items():
            cell.PAD = self.style.spacing("table_pad")
            if r == 0:
                cell.set_facecolor("#f0f0f0")
                cell.set_text_props(weight="bold", fontsize=fontsize)
                continue
            if c == 0:
                cell.set_text_props(fontsize=self.style.fontsize("table"), ha="left")
            elif c == -1:
                cell.set_text_props(fontsize=self.style.fontsize("table"), ha="left")

        # Merge the row-label cell of each method group across its sub-rows.
        # Keep the reader name and selector on one line in the merged label
        # column, matching every other method label in the analysis output.
        for group_idx, (start, end) in enumerate(table.groups):
            lines = row_label_lines[table.row_labels[group_idx]]
            for r in range(start, end + 1):
                cell = mpl_table[r + 1, -1]
                cell.visible_edges = "LR" + ("T" if r == start else "") + ("B" if r == end else "")
                cell.set_facecolor("#eef1f8")
                cell.get_text().set_text(one_line_label(lines) if r == start else "")

        fig.tight_layout(pad=self.style.table_tight_layout_pad)
        save_figure(fig, filepath, self.style)

    @property
    def signature(self) -> Config:
        """Return the statistics table plotter signature.

        Returns:
            Configuration values needed to recreate this plotter.
        """
        signature = super().signature
        signature.update_with_dict(
            {"stat_keys": self.stat_keys, "precision": self.precision, "sort_key": self.sort_key}
        )
        return signature


def _annotate_sort_column(labels: list[str], sort_col: int, max_width: int) -> list[str]:
    """Append the sort indicator to the header of the sorting column.

    Args:
        labels: Column header labels.
        sort_col: Index of the column whose values order the rows.
        max_width: Maximum rendered width of each label.

    Returns:
        Header labels with the sort indicator appended to the sorting column.
    """
    return [
        truncate_text(label, max_width) + (_SORT_INDICATOR if idx == sort_col else "")
        for idx, label in enumerate(labels)
    ]


def _cell_colors(cell_marks: list[list[str | None]]) -> list[list[str]]:
    """Map best/worst cell marks to Matplotlib background colors.

    Args:
        cell_marks: Per-cell marks from ``stat_tables.cell_marks``.

    Returns:
        Matrix of Matplotlib-compatible color strings matching ``cell_marks``.
    """
    return [[_MARK_COLORS[mark] if mark is not None else "white" for mark in marks_row] for marks_row in cell_marks]
