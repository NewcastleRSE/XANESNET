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

from .base import Reporter
from .registry import ReporterRegistry


@ReporterRegistry.register("markdown")
class MarkdownReporter(Reporter):
    """
    Export aggregated results to Markdown format.
    """

    def __init__(self, filename: str = "summary.md"):
        """
        Args:
            filename: Output Markdown filename
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
        Generate Markdown report.
        """
        if not aggregated_results:
            logging.warning("No aggregated results to export")
            return

        output_path = output_dir / self.filename

        with open(output_path, "w") as f:
            # Write header
            f.write("# Analysis Summary Report\n\n")

            # Write sample count
            sample_count = len(per_sample_results)
            f.write(f"**Total Samples Analyzed:** {sample_count}\n\n")

            # Write results (flat structure now)
            f.write("## Metrics\n\n")

            for metric_name, stats in aggregated_results.items():
                f.write(f"### {metric_name.upper()}\n\n")
                f.write("| Statistic | Value |\n")
                f.write("|-----------|-------|\n")

                for stat_name, value in stats.items():
                    f.write(f"| {stat_name} | {value:.6f} |\n")

                f.write("\n")

        logging.info(f"Saved Markdown report to {output_path}")
