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
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from .base import Reporter
from .registry import ReporterRegistry


@ReporterRegistry.register("csv")
class CSVReporter(Reporter):
    """
    Export aggregated results to CSV format.
    """

    def __init__(self, filename: str = "summary.csv"):
        """
        Args:
            filename: Output CSV filename
        """
        self.filename = filename

    def report(
        self,
        selector: Iterable[dict[str, Any]],
        per_sample_results: list[dict[str, Any]],
        aggregated_results: dict[str, Any],
        output_dir: Path,
    ) -> None:
        """
        Generate CSV report.
        """
        if not aggregated_results:
            logging.warning("No aggregated results to export")
            return

        output_path = output_dir / self.filename

        # Flatten structure for CSV
        rows = []
        for metric_name, stats in aggregated_results.items():
            row = {"metric": metric_name}
            row.update(stats)
            rows.append(row)

        if not rows:
            logging.warning("No data to write to CSV")
            return

        # Write CSV
        with open(output_path, "w", newline="") as f:
            fieldnames = rows[0].keys()
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        logging.info(f"Saved CSV report to {output_path}")
