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

"""Base reporter interface and shared reporter helpers."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, cast

from ..result import AnalysisResults


class Reporter(ABC):
    """Base class for analysis reporters.

    Reporters consume analysis results and write machine-readable files such as CSV, JSON, or YAML.

    Args:
        reporter_type: Registered reporter name from the analysis configuration.

    Attributes:
        reporter_type: Registered reporter name from the analysis configuration.
    """

    def __init__(self, reporter_type: str) -> None:
        """Initialize a reporter instance."""
        self.reporter_type = reporter_type

    @abstractmethod
    def report(
        self,
        results: AnalysisResults,
        output_dir: Path,
    ) -> None:
        """Generate report files from analysis results.

        Args:
            results: Analysis pipeline outputs to report.
            output_dir: Directory where report files should be written.
        """
        ...


def selector_label(selectors_config: list[dict[str, Any]], sel_idx: int) -> str:
    """Return the configured selector type label for a selector index.

    Args:
        selectors_config: Selector configuration dictionaries in pipeline order.
        sel_idx: Zero-based selector index.

    Returns:
        Configured ``selector_type`` when present, otherwise ``"unknown"``.
    """
    if sel_idx < len(selectors_config):
        return cast(str, selectors_config[sel_idx].get("selector_type", "unknown"))
    return "unknown"
