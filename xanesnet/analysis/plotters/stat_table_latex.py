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
        time_unit: Display unit for ``time_per_spectrum`` values.
        decimal_places: Fixed number of digits after the decimal point.
        display_scales: Optional display multipliers keyed by scalar value key.
        sort_key: Scalar value key whose first statistic orders the combined
            table rows. ``None`` uses the first value key.
        plot_size: Shared figure size profile: ``"small"`` or ``"default"``.
    """

    def __init__(
        self,
        plotter_type: str,
        stat_keys: list[str],
        sort_key: str | None,
        plot_size: PlotSize,
        time_unit: str,
        decimal_places: int,
        display_scales: Config,
    ) -> None:
        """Initialize a LaTeX statistics table plotter."""
        super().__init__(plotter_type, latex_font=False, plot_size=plot_size)
        self.stat_keys = stat_keys
        self.sort_key = sort_key
        self.time_unit = time_unit
        self.decimal_places = decimal_places
        self.display_scales = {key: float(value) for key, value in display_scales.as_dict().items()}

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
            table = stat_tables.build_single_table(
                rows,
                self.stat_keys,
                self.decimal_places,
                value_key=value_key,
                time_unit=self.time_unit,
                display_scales=self.display_scales,
            )
            if table is None:
                continue
            agg_dir = root / stat_tables.aggregator_dir_name(agg_type, agg_idx)
            agg_dir.mkdir(parents=True, exist_ok=True)
            self._write(agg_dir / f"{value_key}.tex", self._single_table_body(table, value_key))

        for (agg_type, agg_idx), rows in source.combined.items():
            available_value_keys = source.value_order.get((agg_type, agg_idx), [])
            table = stat_tables.build_combined_table(
                rows,
                source.row_order,
                available_value_keys,
                self.stat_keys,
                self.decimal_places,
                self.sort_key,
                time_unit=self.time_unit,
                display_scales=self.display_scales,
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
        display_value_key = stat_tables.value_display_label(value_key, self.time_unit, self.display_scales)
        caption = f"PLACEHOLDER: statistics for '{latex.escape_latex(display_value_key)}'."
        label = f"tab:stat_{latex.sanitize_label(value_key)}"
        time_unit = (
            stat_tables.value_display_unit(value_key, self.time_unit, self.display_scales)
            if value_key == "time_per_spectrum"
            else None
        )
        headers = []
        for idx, label in enumerate(table.col_labels):
            if time_unit is not None:
                label = label.removesuffix(f" ({time_unit})")
            header = latex.escape_latex(truncate_text(label, self.style.table_label_width()))
            header += _SORT_INDICATOR if idx == table.sort_col else ""
            headers.append(latex.format_label_cell(header, padding="0pt"))
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
            cells = [latex.format_label_cell(latex.escape_latex(row_label))]
            for c in range(len(table.col_labels)):
                cells.append(latex.format_cell(table.cell_text[r][c], table.cell_marks[r][c], time_unit))
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
        headers = []
        for idx, value_col in enumerate(table.value_cols):
            label_text = stat_tables.value_display_label(value_col, self.time_unit, self.display_scales)
            time_unit = stat_tables.value_display_unit(value_col, self.time_unit, self.display_scales)
            if value_col == "time_per_spectrum" and time_unit is not None:
                label_text = label_text.removesuffix(f" ({time_unit})")
            header = latex.escape_latex(truncate_text(label_text, self.style.table_label_width()))
            header += _SORT_INDICATOR if idx == table.sort_col else ""
            headers.append(latex.format_label_cell(header, padding="0pt"))
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
            merged_label = latex.escape_latex(one_line_label(label_lines))
            for r in range(start, end + 1):
                merged_cell = rf"\multirow{{{n_sub_rows}}}{{*}}{{{merged_label}}}"
                label_cell = latex.format_label_cell(merged_cell) if r == start else ""
                cells = [label_cell, latex.format_label_cell(latex.escape_latex(table.cell_text[r][0]))]
                for c in range(len(table.value_cols)):
                    value_col = table.value_cols[c]
                    unit = (
                        stat_tables.value_display_unit(value_col, self.time_unit, self.display_scales)
                        if value_col == "time_per_spectrum"
                        else None
                    )
                    cells.append(latex.format_cell(table.cell_text[r][c + 1], table.cell_marks[r][c], unit))
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
                "time_unit": self.time_unit,
                "decimal_places": self.decimal_places,
                "display_scales": self.display_scales,
                "sort_key": self.sort_key,
            }
        )
        return signature
