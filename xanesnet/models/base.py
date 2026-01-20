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

from torch import nn

###############################################################################
#################################### CLASS ####################################
###############################################################################


class Model(nn.Module):
    """
    Abstract base class for models.
    All model classes should inherit from this class and implement the required methods.
    """

    def __init__(
        self,
        model_type: str,
        params: dict,
    ):
        super().__init__()

        self.model_type = model_type
        self.params = params

    @abstractmethod
    def init_weights(self, weights_init, bias_init, **kwargs):
        """
        Initialise model weights and bias.
        This method should be implemented by all subclasses.
        """
        return

    @property
    def signature(self) -> dict:
        """Return model signature as a dictionary."""
        signature = {
            "model_type": self.model_type,
            "params": self.params,
        }
        signature["params"].pop("model_type", None)  # Remove redundant model_type from params
        return signature
