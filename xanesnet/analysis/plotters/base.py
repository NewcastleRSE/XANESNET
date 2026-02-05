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


class Plotter(ABC):
    """
    Base class for plotting modules.
    """

    def __init__(self, plotter_type: str) -> None:
        self.plotter_type = plotter_type

    @abstractmethod
    def plot(
        self,
        selector: Iterable[dict[str, Any]],
        selector_config: dict[str, Any],
        per_sample_results: list[dict[str, Any]],
        per_sample_configs: list[dict[str, Any]],
        aggregated_results: list[dict[str, Any]],
        aggregator_configs: list[dict[str, Any]],
        output_dir: Path,
    ) -> None:
        """
        Generate plots and save to output directory.

        Args:
            selector: Iterable over selected samples from data source
            selector_config: Configuration dict for this selector
            per_sample_results: List of per-sample results for this selection
            per_sample_configs: List of configs for all per-sample modules
            aggregated_results: List of aggregated results for this selection
            aggregator_configs: List of configs for all aggregators
            output_dir: Directory to save plots
        """
        ...
