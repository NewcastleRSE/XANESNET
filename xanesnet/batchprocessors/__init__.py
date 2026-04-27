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

"""Public API for all XANESNET batch processors."""

from .base import BatchProcessor
from .descriptor_mlp import XanesXMLPBatchProcessor
from .e3ee import E3EEBatchProcessor
from .e3ee_full import E3EEFullBatchProcessor
from .envembed import EnvEmbedBatchProcessor
from .gemnet import GemNetBatchProcessor
from .gemnet_oc import GemNetOCBatchProcessor
from .geometrygraph_dimenet import GeometryGraphDimeNetBatchProcessor
from .geometrygraph_schnet import GeometryGraphSchNetBatchProcessor
from .registry import BatchProcessorRegistry

__all__ = [
    "BatchProcessor",
    "BatchProcessorRegistry",
    "XanesXMLPBatchProcessor",
    "GemNetBatchProcessor",
    "GemNetOCBatchProcessor",
    "E3EEBatchProcessor",
    "E3EEFullBatchProcessor",
    "EnvEmbedBatchProcessor",
    "GeometryGraphDimeNetBatchProcessor",
    "GeometryGraphSchNetBatchProcessor",
]
