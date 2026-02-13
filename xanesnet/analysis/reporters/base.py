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

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from ..result import AnalysisResults


class Reporter(ABC):
    """
    Base class for reporters.

    Reporters consume analysis results and write mostly machine-readable
    reports to disk (e.g. CSV, JSON, etc.).
    """

    def __init__(self, reporter_type: str) -> None:
        self.reporter_type = reporter_type

    @abstractmethod
    def report(
        self,
        results: AnalysisResults,
        output_dir: Path,
    ) -> None:
        """
        Generate a report from analysis results and save to output_dir.
        """
        ...


def selector_label(selectors_config: list[dict[str, Any]], sel_idx: int) -> str:
    """
    Derive a human-readable label from the selector config at sel_idx.
    """
    if sel_idx < len(selectors_config):
        return selectors_config[sel_idx].get("selector_type", "unknown")
    return "unknown"
