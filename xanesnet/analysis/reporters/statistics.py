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

"""Reporter that writes aggregated statistics to YAML or JSON files."""

import json
import logging
from pathlib import Path
from typing import Any, ClassVar

import yaml

from ..aggregators import AggregatorResult
from ..result import AnalysisResults
from .base import Reporter, selector_label
from .registry import ReporterRegistry


@ReporterRegistry.register("statistics")
class StatisticsReporter(Reporter):
    """Write aggregated statistics as structured files.

    Produces one file per (selector, predictions_reader, aggregator) combination.
    Each file includes a ``metadata`` section for traceability and a ``statistics``
    section containing the full aggregation output.

    Supported formats: ``yaml`` (default), ``json``.

    Args:
        reporter_type: Registered reporter name from the analysis configuration.
        format: Output format. Supported values are ``"yaml"`` and ``"json"``.
        **kwargs: Accepted for configuration compatibility and ignored.
    """

    SUPPORTED_FORMATS: ClassVar[tuple[str, str]] = ("yaml", "json")

    def __init__(self, reporter_type: str, format: str = "yaml", **kwargs: Any) -> None:
        """Initialize a statistics reporter.

        Raises:
            ValueError: If ``format`` is not one of ``SUPPORTED_FORMATS``.
        """
        super().__init__(reporter_type)
        if format not in self.SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported format '{format}'. Choose from {self.SUPPORTED_FORMATS}")
        self.format = format

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
                sel_label = selector_label(results.selectors_config, sel_idx)
                for agg_result in agg_results:
                    agg_label = f"{agg_result.aggregator_type}_{agg_result.aggregator_index:03d}"
                    filename = (
                        f"pred_{reader_idx:03d}" f"__sel_{sel_idx:03d}_{sel_label}" f"__{agg_label}" f".{self.format}"
                    )
                    filepath = report_dir / filename

                    report = self._build_report(results, sel_idx, reader_idx, agg_result)
                    self._save(report, filepath)

    @staticmethod
    def _build_report(
        results: AnalysisResults,
        sel_idx: int,
        reader_idx: int,
        agg_result: AggregatorResult,
    ) -> dict[str, Any]:
        """Build a self-describing statistics report payload.

        Args:
            results: Analysis pipeline outputs containing configurations.
            sel_idx: Zero-based selector index for this aggregation result.
            reader_idx: Zero-based prediction reader index for this aggregation result.
            agg_result: Aggregation result to serialize.

        Returns:
            Report dictionary with ``metadata`` and ``statistics`` sections.
        """
        sel_cfg = results.selectors_config[sel_idx] if sel_idx < len(results.selectors_config) else {}
        agg_cfg = (
            results.aggregators_config[agg_result.aggregator_index]
            if agg_result.aggregator_index < len(results.aggregators_config)
            else {}
        )

        return {
            "metadata": {
                "predictions_index": reader_idx,
                "selector_index": sel_idx,
                "selector_config": sel_cfg,
                "aggregator_type": agg_result.aggregator_type,
                "aggregator_index": agg_result.aggregator_index,
                "aggregator_config": agg_cfg,
            },
            "statistics": agg_result.data,
        }

    def _save(self, report: dict[str, Any], filepath: Path) -> None:
        """Write a statistics report to disk in the configured format.

        Args:
            report: Report payload produced by ``_build_report``.
            filepath: Destination file path. The suffix should match ``self.format``.
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
