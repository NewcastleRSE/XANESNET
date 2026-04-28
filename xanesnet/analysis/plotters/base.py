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

"""Base plotter interface for analysis visualizations."""

from abc import ABC, abstractmethod
from pathlib import Path

from ..result import AnalysisResults


class Plotter(ABC):
    """Base class for analysis plotters.

    Plotters consume analysis results and write plots to disk.

    Args:
        plotter_type: Registered plotter name from the analysis configuration.

    Attributes:
        plotter_type: Registered plotter name from the analysis configuration.
    """

    def __init__(self, plotter_type: str) -> None:
        """Initialize a plotter instance."""
        self.plotter_type = plotter_type

    @abstractmethod
    def plot(
        self,
        results: AnalysisResults,
        output_dir: Path,
    ) -> None:
        """Generate plot files from analysis results.

        Args:
            results: Analysis pipeline outputs to plot.
            output_dir: Directory where plot files should be written.
        """
        ...
