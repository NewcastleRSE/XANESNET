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

"""Table data assembly helpers."""

import logging
from dataclasses import dataclass
from typing import Any, cast

from ...result import AnalysisResults
from ...utils import is_scalar_value
from .formatting import format_decimal

AggregatorKey = tuple[str, int]
StatValues = dict[str, float]


@dataclass(frozen=True)
class TableSource:
    """Aggregated statistics grouped for table rendering.

    Attributes:
        single: Rows of every single-value table, keyed by aggregator and value key.
        combined: Rows of every combined table, keyed by aggregator.
        row_order: Method row labels in first-seen order.
        value_order: Scalar value keys per aggregator in first-seen order.
        row_label_lines: Stacked label lines per method row label.
    """

    single: dict[tuple[str, int, str], dict[str, StatValues]]
    combined: dict[AggregatorKey, dict[str, dict[str, StatValues]]]
    row_order: list[str]
    value_order: dict[AggregatorKey, list[str]]
    row_label_lines: dict[str, list[str]]


@dataclass(frozen=True)
class SingleTable:
    """Laid-out table comparing methods for one scalar value key.

    Attributes:
        row_labels: Method labels, one per row.
        col_labels: Statistic keys, one per column.
        cell_text: Preformatted cell texts matching ``cell_values``.
        cell_values: Numeric cell values; ``None`` marks a missing statistic.
        cell_marks: Per-cell ``"best"``/``"worst"`` marks, or ``None``.
        sort_col: Index of the column whose values order the rows.
    """

    row_labels: list[str]
    col_labels: list[str]
    cell_text: list[list[str]]
    cell_values: list[list[float | None]]
    cell_marks: list[list[str | None]]
    sort_col: int


@dataclass(frozen=True)
class CombinedTable:
    """Laid-out table comparing methods across all scalar value keys.

    Rows are grouped by method with one sub-row per statistic key.

    Attributes:
        row_labels: Method labels, one per group in ``groups``.
        value_cols: Scalar value keys, one per data column.
        groups: Inclusive sub-row index range ``(start, end)`` of each method.
        cell_text: Cell texts including the leading statistic-name column.
        cell_values: Numeric values of the data columns only; ``None`` marks a
            missing statistic.
        cell_marks: Per-cell marks aligned with ``cell_values``.
        sort_col: Index into ``value_cols`` of the column whose values order
            the rows.
    """

    row_labels: list[str]
    value_cols: list[str]
    groups: list[tuple[int, int]]
    cell_text: list[list[str]]
    cell_values: list[list[float | None]]
    cell_marks: list[list[str | None]]
    sort_col: int


def collect_tables(results: AnalysisResults) -> TableSource:
    """Group the scalar statistics of every aggregator result by method.

    Only aggregator values that are themselves mappings of statistic name to
    number are table data; other values, such as the per-channel curves of the
    vector aggregators, are skipped.

    Args:
        results: Analysis pipeline outputs to read aggregations from.

    Returns:
        Grouped statistics ready to be laid out into tables.
    """
    single: dict[tuple[str, int, str], dict[str, StatValues]] = {}
    combined: dict[AggregatorKey, dict[str, dict[str, StatValues]]] = {}
    row_order: list[str] = []
    value_order: dict[AggregatorKey, list[str]] = {}
    row_label_lines: dict[str, list[str]] = {}

    for reader_idx, reader_results in enumerate(results.aggregator_results):
        logging.info(f"    Predictions {reader_idx + 1}/{len(results.aggregator_results)}.")

        for sel_idx, agg_results in enumerate(reader_results):
            label = results.method_label(reader_idx, sel_idx)
            row_label = label.joined
            if row_label not in row_order:
                row_order.append(row_label)
                row_label_lines[row_label] = label.lines

            for agg_result in agg_results:
                agg_key = (agg_result.aggregator_type, agg_result.aggregator_index)
                for value_key, stats in agg_result.data.items():
                    if not _is_stat_mapping(stats):
                        continue
                    stat_vals = cast(StatValues, stats)
                    single.setdefault((*agg_key, value_key), {})[row_label] = stat_vals
                    combined.setdefault(agg_key, {}).setdefault(row_label, {})[value_key] = stat_vals
                    if value_key not in value_order.setdefault(agg_key, []):
                        value_order[agg_key].append(value_key)

    return TableSource(single, combined, row_order, value_order, row_label_lines)


def _is_stat_mapping(value: Any) -> bool:
    """Return whether an aggregator value is a table-renderable statistics mapping.

    Args:
        value: Value taken from ``AggregatorResult.data``.

    Returns:
        ``True`` for a non-empty mapping from statistic name to scalar number.
        Mappings that carry arrays, such as the per-channel curves of the
        vector aggregator, are rejected.
    """
    if not isinstance(value, dict) or not value:
        return False
    items = cast(dict[Any, Any], value)
    return all(isinstance(key, str) and is_scalar_value(stat) for key, stat in items.items())


def aggregator_dir_name(agg_type: str, agg_idx: int) -> str:
    """Return the output subdirectory name of one aggregator.

    Args:
        agg_type: Registered aggregator name that produced the statistics.
        agg_idx: Zero-based aggregator index from the analysis configuration.

    Returns:
        Directory name below the table output root.
    """
    return f"{agg_type}_{agg_idx:03d}"


def combined_stem(value_keys: list[str]) -> str:
    """Return the file stem of the combined table.

    Args:
        value_keys: Scalar value keys of the aggregator.

    Returns:
        ``"combined"``, or ``"combined_table"`` when a value key already
        occupies that name.
    """
    return "combined" if "combined" not in value_keys else "combined_table"


_VALUE_DISPLAY_LABELS: dict[str, str] = {"time_per_spectrum": "time"}


def value_display_label(value_key: str) -> str:
    """Return the presentation label for a raw scalar value key.

    The raw key remains unchanged for aggregation, sorting, filenames, and
    LaTeX labels; this mapping affects rendered table text only.

    Args:
        value_key: Raw scalar value key.

    Returns:
        Display label for the value key.
    """
    return _VALUE_DISPLAY_LABELS.get(value_key, value_key)


def _order_rank(value: float | None) -> tuple[int, float]:
    """Return a sort key that places missing values after numeric values.

    Args:
        value: Numeric cell value, or ``None`` when missing.

    Returns:
        ``(rank, value)`` where present values rank before missing ones.
    """
    if value is None:
        return (1, 0.0)
    return (0, float(value))


def _first_stat(row_values: dict[str, StatValues], value_col: str, stat_key: str | None) -> float | None:
    """Return the ordering statistic of one method row.

    Args:
        row_values: Mapping from value key to statistic values of one method.
        value_col: Value key of the first table column.
        stat_key: Configured statistic key used for ordering, or ``None`` when
            no ordering statistic is available.

    Returns:
        The statistic value used for row ordering, or ``None`` when absent.
    """
    stats = row_values.get(value_col, {})
    if not stats or stat_key is None:
        return None
    return stats.get(stat_key)


def build_single_table(
    rows: dict[str, StatValues],
    stat_keys: list[str],
    precision: int,
) -> SingleTable | None:
    """Lay out one method comparison table for a single scalar value key.

    Args:
        rows: Mapping from method row label to its statistic values.
        stat_keys: Ordered statistic keys used as columns; keys absent from the
            data are dropped.
        precision: Number of significant digits used to format the cell texts.

    Returns:
        Laid-out table, or ``None`` when it would have no rows or columns.
    """
    row_labels = list(rows.keys())
    col_labels = [key for key in stat_keys if any(key in stats for stats in rows.values())]
    if not col_labels or not row_labels:
        return None

    # Order rows by the first column so the best value sits at the top and the
    # worst at the bottom; missing values sort last.
    row_labels.sort(key=lambda label: _order_rank(rows[label].get(col_labels[0])))

    cell_text: list[list[str]] = []
    cell_values: list[list[float | None]] = []
    for row_label in row_labels:
        stats = rows[row_label]
        text_row, value_row = _format_row([stats.get(key) for key in col_labels], precision)
        cell_text.append(text_row)
        cell_values.append(value_row)

    return SingleTable(row_labels, col_labels, cell_text, cell_values, cell_marks(cell_values), sort_col=0)


def build_combined_table(
    rows: dict[str, dict[str, StatValues]],
    row_order: list[str],
    value_keys: list[str],
    stat_keys: list[str],
    precision: int,
    sort_key: str | None = None,
) -> CombinedTable | None:
    """Lay out the combined table covering all scalar value keys of one aggregator.

    Args:
        rows: Mapping from method row label to value key to statistic values.
        row_order: Method row labels in first-seen order.
        value_keys: Scalar value keys in first-seen order.
        stat_keys: Ordered statistic keys used as sub-rows; keys absent from the
            data are dropped.
        precision: Number of significant digits used to format the cell texts.
        sort_key: Scalar value key whose first statistic orders the rows. When
            ``None`` or absent from the data, the first value key is used.

    Returns:
        Laid-out table, or ``None`` when it would have no rows or columns.
    """
    requested_stats = set(stat_keys)
    value_cols = [
        key
        for key in value_keys
        if any(requested_stats.intersection(rows[row].get(key, {})) for row in row_order if row in rows)
    ]
    row_labels = [
        row
        for row in row_order
        if row in rows
        and any(
            any(stat_key in rows[row].get(value_key, {}) for stat_key in requested_stats) for value_key in value_cols
        )
    ]
    if not value_cols or not row_labels:
        return None

    # Order methods by the first statistic of the configured sort key, falling
    # back to the first value column when no sort key is given. The best value
    # sits at the top and the worst at the bottom.
    sort_value_col = sort_key if sort_key is not None and sort_key in value_cols else value_cols[0]
    sort_col = value_cols.index(sort_value_col)
    first_stat_key = stat_keys[0] if stat_keys else None
    row_labels.sort(key=lambda label: _order_rank(_first_stat(rows[label], sort_value_col, first_stat_key)))

    used_stat_keys = [
        key
        for key in stat_keys
        if any(key in rows[row].get(value_key, {}) for row in row_labels for value_key in value_cols)
    ]
    if not used_stat_keys:
        return None

    cell_text: list[list[str]] = []
    cell_values: list[list[float | None]] = []
    row_stats: list[str] = []
    groups: list[tuple[int, int]] = []

    def add_sub_row(row_label: str, stat_key: str) -> None:
        """Append one sub-row for a method and statistic key.

        Args:
            row_label: Method label whose values should be added.
            stat_key: Statistic key to add as the sub-row label.
        """
        raw = [rows[row_label].get(value_key, {}).get(stat_key) for value_key in value_cols]
        text_row, value_row = _format_row(raw, precision)
        cell_text.append([stat_key, *text_row])
        cell_values.append(value_row)
        row_stats.append(stat_key)

    for row_label in row_labels:
        start = len(cell_text)
        for stat_key in used_stat_keys:
            if any(stat_key in rows[row_label].get(value_key, {}) for value_key in value_cols):
                add_sub_row(row_label, stat_key)
        if len(cell_text) == start:
            continue
        groups.append((start, len(cell_text) - 1))

    marks = cell_marks(cell_values, row_groups=row_stats)
    return CombinedTable(row_labels, value_cols, groups, cell_text, cell_values, marks, sort_col=sort_col)


def cell_marks(
    cell_values: list[list[float | None]],
    row_groups: list[Any] | None = None,
) -> list[list[str | None]]:
    """Determine per-cell best/worst marks for highlighting.

    Within each column, the best (lowest) and worst (highest) values are
    marked. When ``row_groups`` is given, best and worst are determined among
    rows that share the same group key, so only comparable rows are compared
    (for example one group per statistic key).

    Args:
        cell_values: Numeric table values with missing values represented by ``None``.
        row_groups: Optional per-row group keys; rows with the same key are
            compared against each other within each column.

    Returns:
        Matrix of ``"best"``, ``"worst"``, or ``None`` entries matching ``cell_values``.
    """
    n_rows = len(cell_values)
    n_cols = len(cell_values[0]) if cell_values else 0
    marks: list[list[str | None]] = [[None] * n_cols for _ in range(n_rows)]

    if n_rows < 2:
        return marks

    groups = row_groups if row_groups is not None else [0] * n_rows

    for c in range(n_cols):
        for group_key in sorted(set(groups), key=str):
            col_vals: list[tuple[int, float]] = []
            for r in range(n_rows):
                if groups[r] != group_key:
                    continue
                value = cell_values[r][c]
                if value is not None:
                    col_vals.append((r, value))
            if len(col_vals) < 2:
                continue
            sorted_vals = sorted(col_vals, key=lambda item: item[1])
            marks[sorted_vals[0][0]][c] = "best"
            marks[sorted_vals[-1][0]][c] = "worst"

    return marks


def _format_row(values: list[float | None], precision: int) -> tuple[list[str], list[float | None]]:
    """Format one row of raw statistic values.

    Args:
        values: Raw values in column order; ``None`` marks a missing statistic.
        precision: Number of significant digits used to format the cell texts.

    Returns:
        ``(texts, values)`` where missing values render as a dash.
    """
    texts = [format_decimal(value, precision) if value is not None else "-" for value in values]
    return texts, list(values)
