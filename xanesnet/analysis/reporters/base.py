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
from collections.abc import Iterable
from pathlib import Path
from typing import Any


class Reporter(ABC):
    """
    Base class for report generation modules.

    Reporters export analysis results to various formats.
    """

    @abstractmethod
    def report(
        self,
        selector: Iterable[dict[str, Any]],
        per_sample_results: list[dict[str, Any]],
        aggregated_results: dict[str, Any],
        output_dir: Path,
    ) -> None:
        """
        Generate reports and save to output directory.

        Args:
            selector: Iterable over selected samples from data source. Can be iterated multiple times
                     if needed to access original sample data.
            per_sample_results: List of per-sample results dictionaries, each containing values collected
                               by per-sample modules. Each dict includes 'sample_id' for traceability.
            aggregated_results: Dictionary containing aggregated statistics across samples.
            output_dir: Directory to save reports.
        """
        ...
