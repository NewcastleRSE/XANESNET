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

"""Analysis pipeline registries and component package exports."""

from .aggregators import AggregatorRegistry
from .collectors import CollectorRegistry
from .plotters import PlotterRegistry
from .reporters import ReporterRegistry
from .selectors import SelectorRegistry

__all__ = [
    "SelectorRegistry",
    "CollectorRegistry",
    "AggregatorRegistry",
    "PlotterRegistry",
    "ReporterRegistry",
]
