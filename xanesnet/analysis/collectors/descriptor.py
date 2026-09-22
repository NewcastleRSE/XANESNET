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

"""Collector that computes a structure descriptor vector per sample."""

from typing import Any

from xanesnet.serialization.config import Config
from xanesnet.serialization.prediction_readers import PredictionSample

from ..descriptor_cache import descriptor_vector
from .base import Collector
from .registry import CollectorRegistry


@CollectorRegistry.register("descriptor")
class DescriptorCollector(Collector):
    """Compute the configured structure descriptor for one sample.

    The descriptor vector of the sample's structure (centered on the target
    site when available) is returned under the ``descriptor`` key.

    Requires:
        Matched raw structures: provided by a structure-matched prediction reader.

    Args:
        collector_type: Registered collector name from the analysis configuration.
        descriptor: Descriptor configuration object used to embed each structure.
    """

    def __init__(
        self,
        collector_type: str,
        descriptor: Config,
    ) -> None:
        """Initialize a descriptor collector."""
        super().__init__(collector_type)
        self.descriptor_config = descriptor

    def process(self, sample: PredictionSample) -> dict[str, Any]:
        """Compute the descriptor vector for one prediction sample.

        Args:
            sample: Prediction sample carrying a matched raw ``structure``.

        Returns:
            Mapping with the descriptor vector under the ``descriptor`` key.
        """
        return {"descriptor": descriptor_vector(self.descriptor_config, sample).tolist()}

    @property
    def signature(self) -> Config:
        """Return the collector signature.

        Returns:
            Configuration values needed to recreate this collector.
        """
        signature = super().signature
        signature.update_with_dict({"descriptor": self.descriptor_config.as_dict()})
        return signature
