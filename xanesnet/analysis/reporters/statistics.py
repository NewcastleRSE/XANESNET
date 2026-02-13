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

import json
import logging
from pathlib import Path
from typing import Any

import yaml

from ..aggregators import AggregatorResult
from ..result import AnalysisResults
from .base import Reporter, selector_label
from .registry import ReporterRegistry


@ReporterRegistry.register("statistics")
class StatisticsReporter(Reporter):
    """
    Reports aggregated statistics (e.g. from ScalarAggregator) as structured files.

    Produces one file per (selector, predictions_reader, aggregator) combination.
    Each file includes a ``metadata`` section for traceability and a ``statistics``
    section containing the full aggregation output.

    Supported formats: ``yaml`` (default), ``json``.

    File naming:
        ``pred_{reader_idx}__sel_{sel_idx}_{selector_type}__{aggregator_type}_{agg_idx}.{format}``
    """

    SUPPORTED_FORMATS = ("yaml", "json")

    def __init__(self, reporter_type: str, format: str = "yaml", **kwargs: Any) -> None:
        super().__init__(reporter_type)
        if format not in self.SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported format '{format}'. Choose from {self.SUPPORTED_FORMATS}")
        self.format = format

    def report(self, results: AnalysisResults, output_dir: Path) -> None:
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
        """
        Build a self-describing report dict with metadata and statistics.
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
        """
        Write report to disk in the configured format.
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
