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

from abc import abstractmethod
from typing import Any

from torch import nn

###############################################################################
#################################### CLASS ####################################
###############################################################################


class Model(nn.Module):
    """
    Abstract base class for models.
    """

    def __init__(
        self,
        model_type: str,
    ) -> None:
        super().__init__()

        self.model_type = model_type

    @abstractmethod
    def init_weights(self, weights_init: str, bias_init: str, **kwargs) -> None:
        """
        Initialise model weights and bias.
        """
        ...

    @property
    @abstractmethod
    def signature(self) -> dict[str, Any]:
        """
        Return model signature as a dictionary.
        """
        signature = {
            "model_type": self.model_type,
        }
        return signature
