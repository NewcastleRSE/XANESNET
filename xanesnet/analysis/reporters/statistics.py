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

"""Reporter that writes aggregated statistics to YAML or JSON files."""

import json
import logging
from pathlib import Path
from typing import Any

import yaml

from xanesnet.serialization.config import Config
from xanesnet.serialization.jsonl_stream import json_friendly

from ..aggregators import AggregatorResult
from ..result import AnalysisResults
from .base import Reporter
from .registry import ReporterRegistry


@ReporterRegistry.register("statistics")
class StatisticsReporter(Reporter):
    """Write aggregated statistics as structured files.

    Produces one file per (selector, prediction reader, aggregator)
    combination, with a ``metadata`` section and a ``statistics`` section.

    Requires:
        Aggregated statistics: provided by at least one aggregator.

    Args:
        reporter_type: Registered reporter name from the analysis configuration.
        format: Output format. Supported values are ``"yaml"`` and ``"json"``.
        aggregator_types: Registered aggregator names to report. ``None``
            reports every configured aggregator; a list restricts the report to
            those types, which is useful when other aggregators only exist to
            feed a plotter.
    """

    def __init__(
        self,
        reporter_type: str,
        format: str,
        aggregator_types: list[str] | None,
    ) -> None:
        """Initialize a statistics reporter."""
        super().__init__(reporter_type)
        self.format = format
        self.aggregator_types = aggregator_types

    def report(self, results: AnalysisResults, output_dir: Path) -> None:
        """Write one statistics file per aggregation result.

        Args:
            results: Analysis pipeline outputs to report.
            output_dir: Directory where the ``statistics`` report tree should be written.
        """
        if not results.aggregator_results:
            logging.info("    No aggregator results to report.")
            return

        report_dir = output_dir / "statistics"
        report_dir.mkdir(parents=True, exist_ok=True)

        for reader_idx, reader_results in enumerate(results.aggregator_results):
            logging.info(f"    Predictions {reader_idx + 1}/{len(results.aggregator_results)}.")

            for sel_idx, agg_results in enumerate(reader_results):
                dir_name = results.method_label(reader_idx, sel_idx).dir_name
                for agg_result in agg_results:
                    if self.aggregator_types is not None and agg_result.aggregator_type not in self.aggregator_types:
                        continue
                    agg_label = f"{agg_result.aggregator_type}_{agg_result.aggregator_index:03d}"
                    filepath = report_dir / f"{dir_name}__{agg_label}.{self.format}"

                    report = self._build_report(sel_idx, reader_idx, agg_result)
                    self._save(report, filepath)

    @staticmethod
    def _build_report(
        sel_idx: int,
        reader_idx: int,
        agg_result: AggregatorResult,
    ) -> dict[str, Any]:
        """Build a statistics report payload with identifying metadata.

        Args:
            sel_idx: Zero-based selector index for this aggregation result.
            reader_idx: Zero-based prediction reader index for this aggregation result.
            agg_result: Aggregation result to serialize.

        Returns:
            Report dictionary with ``metadata`` and ``statistics`` sections.
        """
        return {
            "metadata": {
                "predictions_index": reader_idx,
                "selector_index": sel_idx,
                "aggregator_type": agg_result.aggregator_type,
                "aggregator_index": agg_result.aggregator_index,
            },
            "statistics": json_friendly(agg_result.data),
        }

    def _save(self, report: dict[str, Any], filepath: Path) -> None:
        """Write a statistics report to disk in the configured format.

        Args:
            report: Report payload produced by ``_build_report``.
            filepath: Destination file path.
        """
        with open(filepath, "w") as f:
            if self.format == "yaml":
                yaml.dump(
                    report,
                    f,
                    default_flow_style=False,
                    sort_keys=False,
                    allow_unicode=True,
                )
            elif self.format == "json":
                json.dump(report, f, indent=2)

    @property
    def signature(self) -> Config:
        """Return the statistics reporter signature.

        Returns:
            Configuration values needed to recreate this reporter.
        """
        signature = super().signature
        signature.update_with_dict({"format": self.format, "aggregator_types": self.aggregator_types})
        return signature
