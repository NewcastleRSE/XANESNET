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

"""Plotter that renders aggregated statistics as publication-ready LaTeX tables."""

import logging
from pathlib import Path

from xanesnet.serialization.config import Config

from ..result import AnalysisResults
from ..utils import one_line_label
from .base import Plotter
from .common import latex, stat_tables
from .common.formatting import truncate_text
from .common.style import PlotSize
from .registry import PlotterRegistry

# Marker appended to the header of the column whose values order the rows.
_SORT_INDICATOR = r" $\downarrow$"


@PlotterRegistry.register("stat_table_latex")
class StatTableLatexPlotter(Plotter):
    """Render comparison tables of aggregated statistics as LaTeX sources.

    Mirrors ``stat_table``; each table is written as a standalone ``.tex``
    document and compiled to PDF when a LaTeX installation is available.

    Requires:
        Scalar statistics: provided by at least one aggregator.

    Args:
        plotter_type: Registered plotter name from the analysis configuration.
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
        plot_size: PlotSize,
    ) -> None:
        """Initialize a LaTeX statistics table plotter."""
        super().__init__(plotter_type, latex_font=False, plot_size=plot_size)
        self.stat_keys = stat_keys
        self.precision = precision
        self.sort_key = sort_key

    def _plot(self, results: AnalysisResults, output_dir: Path) -> None:
        """Write LaTeX sources and compiled PDFs for aggregated scalar statistics.

        Args:
            results: Analysis pipeline outputs to plot.
            output_dir: Directory where the ``stat_tables_latex`` tree should be written.
        """
        if not results.aggregator_results:
            logging.info("    No aggregator results available, skipping.")
            return

        root = output_dir / "stat_tables_latex"
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
            self._write(agg_dir / f"{value_key}.tex", self._single_table_body(table, value_key))

        for (agg_type, agg_idx), rows in source.combined.items():
            available_value_keys = source.value_order.get((agg_type, agg_idx), [])
            table = stat_tables.build_combined_table(
                rows, source.row_order, available_value_keys, self.stat_keys, self.precision, self.sort_key
            )
            if table is None:
                continue
            agg_dir = root / stat_tables.aggregator_dir_name(agg_type, agg_idx)
            agg_dir.mkdir(parents=True, exist_ok=True)
            stem = stat_tables.combined_stem(available_value_keys)
            self._write(agg_dir / f"{stem}.tex", self._combined_table_body(table, source.row_label_lines))

        logging.info(f"    Wrote LaTeX tables to '{root}'.")

    def _single_table_body(self, table: stat_tables.SingleTable, value_key: str) -> str:
        """Build the LaTeX table float for a single-value statistics table.

        Rows are the methods and columns the statistic keys; best and worst
        cells are highlighted with ``cellcolor`` and wrapped in a top-float
        table with a compact publication font.

        Args:
            table: Laid-out single-value table.
            value_key: Scalar value key represented by the table.

        Returns:
            Complete ``table`` float as a LaTeX string.
        """
        col_spec = "l" + "c" * len(table.col_labels)
        display_value_key = stat_tables.value_display_label(value_key)
        caption = f"PLACEHOLDER: statistics for '{latex.escape_latex(display_value_key)}'."
        label = f"tab:stat_{latex.sanitize_label(value_key)}"
        headers = [
            latex.escape_latex(truncate_text(label, self.style.table_label_width()))
            + (_SORT_INDICATOR if idx == table.sort_col else "")
            for idx, label in enumerate(table.col_labels)
        ]
        lines: list[str] = [
            r"\begin{table}[t]",
            rf"\caption{{{caption}}}",
            rf"\label{{{label}}}",
            latex.LATEX_TABLE_FONT,
            rf"\begin{{tabular}}{{{col_spec}}}",
            r"\hline",
            " & ".join(["", *headers]) + r" \\",
            r"\hline",
        ]
        for r, row_label in enumerate(table.row_labels):
            cells = [latex.escape_label_line(row_label)]
            for c in range(len(table.col_labels)):
                cells.append(latex.format_cell(table.cell_values[r][c], table.cell_marks[r][c], self.precision))
            lines.append(" & ".join(cells) + r" \\")
        lines.extend(
            [
                r"\hline",
                r"\end{tabular}",
                r"\end{table}",
            ]
        )
        return "\n".join(lines)

    def _combined_table_body(
        self,
        table: stat_tables.CombinedTable,
        row_label_lines: dict[str, list[str]],
    ) -> str:
        """Build the LaTeX table float for the combined statistics table.

        Rows are grouped by method with one sub-row per statistic key; the
        method label is merged across its sub-rows with ``multirow`` and a
        left-aligned ``shortstack``. The table is wrapped in a top-float
        table with placeholder caption and label text.

        Args:
            table: Laid-out combined table.
            row_label_lines: Label lines per method row label.

        Returns:
            Complete ``table`` float as a LaTeX string.
        """
        caption = "PLACEHOLDER: combined statistics table."
        label = "tab:stat_combined"
        headers = [
            latex.escape_latex(
                truncate_text(
                    stat_tables.value_display_label(value_col),
                    self.style.table_label_width(),
                )
            )
            + (_SORT_INDICATOR if idx == table.sort_col else "")
            for idx, value_col in enumerate(table.value_cols)
        ]
        lines: list[str] = [
            r"\begin{table}[t]",
            rf"\caption{{{caption}}}",
            rf"\label{{{label}}}",
            latex.LATEX_TABLE_FONT,
            rf"\begin{{tabular}}{{ll{'c' * len(table.value_cols)}}}",
            r"\hline",
            " & & " + " & ".join(headers) + r" \\",
            r"\hline",
        ]
        for group_idx, (start, end) in enumerate(table.groups):
            if start > end:
                continue
            n_sub_rows = end - start + 1
            label_lines = row_label_lines[table.row_labels[group_idx]]
            merged_label = latex.escape_label_line(one_line_label(label_lines))
            for r in range(start, end + 1):
                label_cell = rf"\multirow{{{n_sub_rows}}}{{*}}{{{merged_label}}}" if r == start else ""
                cells = [label_cell, latex.escape_latex(table.cell_text[r][0])]
                for c in range(len(table.value_cols)):
                    cells.append(latex.format_cell(table.cell_values[r][c], table.cell_marks[r][c], self.precision))
                lines.append(" & ".join(cells) + r" \\")
            lines.append(r"\hline")
        lines.extend(
            [
                r"\end{tabular}",
                r"\end{table}",
            ]
        )
        return "\n".join(lines)

    @staticmethod
    def _write(tex_path: Path, body: str) -> None:
        """Write a standalone LaTeX document and compile it when possible.

        Args:
            tex_path: Destination ``.tex`` path.
            body: LaTeX table float for the table.
        """
        latex.write_document(tex_path, body)
        latex.compile_pdf(tex_path, tex_path.with_name(f"{tex_path.stem}_latex.pdf"))

    @property
    def signature(self) -> Config:
        """Return the LaTeX statistics table plotter signature.

        Returns:
            Configuration values needed to recreate this plotter.
        """
        signature = super().signature
        signature.update_with_dict(
            {
                "stat_keys": self.stat_keys,
                "precision": self.precision,
                "sort_key": self.sort_key,
            }
        )
        return signature
