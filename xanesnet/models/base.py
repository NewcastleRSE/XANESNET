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

"""Abstract base class for all XANESNET models."""

from abc import ABC, abstractmethod

from torch import nn

from xanesnet.serialization.config import Config

###############################################################################
#################################### CLASS ####################################
###############################################################################


class Model(nn.Module, ABC):
    """Abstract base class for all XANESNET models.

    Subclasses must implement ``init_weights`` and ``signature``.

    Args:
        model_type: String identifier for the model type.
    """

    def __init__(
        self,
        model_type: str,
    ) -> None:
        """Initialize ``Model``."""
        super().__init__()

        self.model_type = model_type

    @abstractmethod
    def init_weights(self, weights_init: str, bias_init: str, **kwargs) -> None:
        """Initialize all model weights and biases.

        Args:
            weights_init: Name of the weight initialization scheme.
            bias_init: Name of the bias initialization scheme.
            **kwargs: Extra keyword arguments forwarded to the weight initializer.
        """
        ...

    @property
    @abstractmethod
    def signature(self) -> Config:
        """Return the model signature as a :class:`~xanesnet.serialization.config.Config`."""
        signature = Config(
            {
                "model_type": self.model_type,
            }
        )
        return signature
