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

"""Base plotter interface for analysis visualizations."""

from abc import ABC, abstractmethod
from pathlib import Path

from xanesnet.serialization.config import Config

from ..result import AnalysisResults
from .common.style import PlotSize, PlotStyle, get_plot_style


class Plotter(ABC):
    """Base class for analysis plotters.

    Plotters consume analysis results and write plots to disk.

    Args:
        plotter_type: Registered plotter name from the analysis configuration.
        latex_font: Render figures in a LaTeX-style serif font when ``True``.
        plot_size: Shared figure size profile: ``"small"`` or ``"default"``.

    Attributes:
        plotter_type: Registered plotter name from the analysis configuration.
        latex_font: Whether figures use the LaTeX-style serif font.
        plot_size: Name of the shared figure size profile.
        style: Resolved immutable rendering style for this plotter.
    """

    def __init__(self, plotter_type: str, latex_font: bool, plot_size: PlotSize) -> None:
        """Initialize a plotter instance."""
        self.plotter_type = plotter_type
        self.latex_font = latex_font
        self.plot_size = plot_size
        self.style: PlotStyle = get_plot_style(plot_size)

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
        with self.style.context(self.latex_font):
            self._plot(results, output_dir)

    @abstractmethod
    def _plot(
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

    @property
    def signature(self) -> Config:
        """Return the plotter signature.

        Returns:
            Configuration values needed to recreate this plotter.
        """
        return Config(
            {
                "plotter_type": self.plotter_type,
                "latex_font": self.latex_font,
                "plot_size": self.plot_size,
            }
        )

    def __str__(self) -> str:
        """Return the short display label of this plotter."""
        return self.plotter_type

    def __repr__(self) -> str:
        """Return a detailed representation of this plotter."""
        args = ", ".join(f"{key}={value!r}" for key, value in self.signature.as_dict().items())
        return f"{type(self).__name__}({args})"
