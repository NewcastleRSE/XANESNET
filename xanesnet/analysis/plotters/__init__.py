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

"""Plotter implementations and registry exports."""

from .base import Plotter
from .bias_variance import BiasVariancePlotter
from .energy_resolved import EnergyResolvedLossPlotter
from .error_correlation import ErrorCorrelationPlotter
from .mean_spectrum import MeanSpectrumPlotter
from .parity import ParityPlotter
from .pca import PcaPlotter
from .registry import PlotterRegistry
from .scalar import ScalarPlotter
from .selector_overview import SelectorOverviewPlotter
from .spectra import AllSpectraPlotter
from .spectra_comparison import SpectraComparisonPlotter
from .stat_table import StatTablePlotter
from .stat_table_latex import StatTableLatexPlotter

__all__ = [
    "Plotter",
    "PlotterRegistry",
    "ScalarPlotter",
    "AllSpectraPlotter",
    "SpectraComparisonPlotter",
    "StatTablePlotter",
    "StatTableLatexPlotter",
    "EnergyResolvedLossPlotter",
    "MeanSpectrumPlotter",
    "ParityPlotter",
    "BiasVariancePlotter",
    "ErrorCorrelationPlotter",
    "PcaPlotter",
    "SelectorOverviewPlotter",
]
