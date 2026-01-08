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

import random
from abc import abstractmethod

import torch
from torch import nn

from xanesnet.utils.switch import BiasInitSwitch, KernelInitSwitch

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
        type: str,
        params: dict,
    ):
        super().__init__()

        self.type = type
        self.params = params

    def init_model_weights(self, kernel: str, bias: str, **kwargs):
        """
        Initialise model kernel and bias weights using user-defined methods.
        """
        if seed is None:
            seed = random.randrange(1000)

        # Set random seed
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
        else:
            torch.manual_seed(seed)

        kernel_init_fn = KernelInitSwitch().get(kernel, **kwargs)
        bias_init_fn = BiasInitSwitch().get(bias)

        # Apply initialisation to each applicable layer
        self.apply(lambda m: self.init_layer_weights(m, kernel_init_fn, bias_init_fn))

    def init_layer_weights(self, m, kernel_init_fn, bias_init_fn):
        """
        Initialise weights and bias for a single layer.
        Function to be overridden by child classes if different layers are used.
        """
        if isinstance(m, (nn.Linear, nn.Conv1d, nn.ConvTranspose1d)):
            kernel_init_fn(m.weight)
            bias_init_fn(m.bias)

    @property
    @abstractmethod
    def metadata(self) -> dict:
        """Return model metadata as a dictionary."""
        metadata = {
            "type": self.type,
            "params": self.params,
        }
        return metadata
