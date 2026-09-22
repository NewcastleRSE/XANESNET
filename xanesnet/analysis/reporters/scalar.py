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

"""Reporter that writes scalar per-sample values to CSV files."""

import csv
import logging
from itertools import repeat
from pathlib import Path

from xanesnet.serialization.jsonl_stream import JSONLStream

from ..result import AnalysisResults
from ..selectors import Selector
from ..utils import ScalarValue, iter_scalar_items, sample_key
from .base import Reporter
from .registry import ReporterRegistry


@ReporterRegistry.register("scalar")
class ScalarReporter(Reporter):
    """Write per-sample scalar values from selectors and collectors as CSV files.

    Requires:
        Scalar collector output (optional).

    Args:
        reporter_type: Registered reporter name from the analysis configuration.
    """

    def __init__(self, reporter_type: str) -> None:
        """Initialize a scalar CSV reporter."""
        super().__init__(reporter_type)

    def report(self, results: AnalysisResults, output_dir: Path) -> None:
        """Write scalar value CSV files grouped by prediction reader and selector.

        Args:
            results: Analysis pipeline outputs to report.
            output_dir: Directory where the ``scalar_values`` report tree should be written.
        """
        if not results.selectors and not results.collector_results:
            logging.info("    No data to report.")
            return

        root = output_dir / "scalar_values"

        for reader_idx, reader_selectors in enumerate(results.selectors):
            logging.info(f"    Predictions {reader_idx + 1}/{len(results.selectors)}.")

            for sel_idx, selector in enumerate(reader_selectors):
                logging.info(f"      Selector {sel_idx + 1}/{len(reader_selectors)}.")
                subdir = root / results.method_label(reader_idx, sel_idx).dir_name
                subdir.mkdir(parents=True, exist_ok=True)

                stream = results.collector_stream(reader_idx, sel_idx)

                self._write_scalar_csvs(selector, stream, subdir)

    @staticmethod
    def _write_scalar_csvs(
        selector: Selector,
        stream: JSONLStream | None,
        output_dir: Path,
    ) -> None:
        """Write one CSV per scalar field found in selected samples and collector values.

        Each CSV uses ``sample_id`` and ``target_site_index`` as identity
        columns followed by one scalar field.

        Args:
            selector: Selector over prediction samples for one prediction reader and selector pair.
            stream: Optional collector result stream aligned with ``selector``.
            output_dir: Directory where CSV files should be written.
        """
        rows_by_key: dict[str, list[tuple[str, int | None, ScalarValue]]] = {}

        for sample, record in zip(selector, stream if stream is not None else repeat({})):
            sample_id, target_site_index = sample_key(sample)
            scalars = dict(iter_scalar_items(sample))
            scalars.update(iter_scalar_items(record))
            for key, value in scalars.items():
                rows_by_key.setdefault(key, []).append((sample_id, target_site_index, value))

        if not rows_by_key:
            logging.info("      No scalar data found, skipping.")
            return

        for key, rows in rows_by_key.items():
            filepath = output_dir / f"{key}.csv"
            with open(filepath, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["sample_id", "target_site_index", key])
                writer.writerows(rows)
