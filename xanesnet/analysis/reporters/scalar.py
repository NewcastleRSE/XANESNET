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

import csv
import logging
from pathlib import Path
from typing import Any, cast

from xanesnet.serialization.jsonl_stream import JSONLStream

from ..selectors import Selector
from .base import AnalysisResults, Reporter, selector_label
from .registry import ReporterRegistry


def _is_scalar(value: Any) -> bool:
    """
    Check if a value is a scalar number (int or float, excluding bool).
    """
    if isinstance(value, bool):
        return False
    return isinstance(value, (int, float))


@ReporterRegistry.register("scalar")
class ScalarReporter(Reporter):
    """
    Reports per-sample scalar values (int, float) from collector results as CSV files.
    """

    def __init__(self, reporter_type: str) -> None:
        super().__init__(reporter_type)

    def report(self, results: AnalysisResults, output_dir: Path) -> None:
        if not results.selectors and not results.collector_results:
            logging.info("    No data to report.")
            return

        root = output_dir / "scalar_values"

        for reader_idx, reader_selectors in enumerate(results.selectors):
            logging.info(f"    Predictions {reader_idx + 1}/{len(results.selectors)}.")

            for sel_idx, selector in enumerate(reader_selectors):
                logging.info(f"      Selector {sel_idx + 1}/{len(reader_selectors)}.")
                sel_label = selector_label(results.selectors_config, sel_idx)
                subdir = root / f"pred_{reader_idx:03d}__sel_{sel_idx:03d}_{sel_label}"
                subdir.mkdir(parents=True, exist_ok=True)

                # Get corresponding collector stream if available
                stream: JSONLStream | None = None
                if reader_idx < len(results.collector_results) and sel_idx < len(results.collector_results[reader_idx]):
                    stream = results.collector_results[reader_idx][sel_idx]

                self._write_scalar_csvs(selector, stream, subdir)

    @staticmethod
    def _write_scalar_csvs(
        selector: Selector,
        stream: JSONLStream | None,
        output_dir: Path,
    ) -> None:
        """
        Iterate selector samples and (optionally) collected values in parallel,
        extract all scalar fields from both, and write one CSV per key.
        """
        rows_by_key: dict[str, list[tuple[Any, float]]] = {}

        if stream is not None:
            for sel_sample, col_sample in zip(selector, stream):
                sid = sel_sample.get("sample_id", col_sample.get("sample_id"))
                for key, value in sel_sample.items():
                    if key != "sample_id" and _is_scalar(value):
                        rows_by_key.setdefault(key, []).append((sid, cast(float, value)))
                for key, value in col_sample.items():
                    if key != "sample_id" and _is_scalar(value):
                        rows_by_key.setdefault(key, []).append((sid, cast(float, value)))
        else:
            for sel_sample in selector:
                sid = sel_sample.get("sample_id")
                for key, value in sel_sample.items():
                    if key != "sample_id" and _is_scalar(value):
                        rows_by_key.setdefault(key, []).append((sid, cast(float, value)))

        if not rows_by_key:
            logging.info(f"      No scalar data found, skipping.")
            return

        for key, rows in rows_by_key.items():
            filepath = output_dir / f"{key}.csv"
            with open(filepath, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["sample_id", key])
                writer.writerows(rows)
