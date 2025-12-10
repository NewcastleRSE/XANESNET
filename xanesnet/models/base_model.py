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

import torch
from torch import nn

from xanesnet.utils.switch import KernelInitSwitch, BiasInitSwitch


class Model(nn.Module):
    """
    Abstract base class for XANESNET models.
    """

    def __init__(self):
        super().__init__()

        self.nn_flag = 0
        self.ae_flag = 0
        self.aegan_flag = 0
        self.gnn_flag = 0

        # 0 if forward() accepts a tensor x, 1 if forward() accepts a batch object
        self.batch_flag = 0
        self.config = {}

    def register_config(self, args, **kwargs):
        """
        Assign arguments from the child class's constructor to self.config.

        Args:
            args: The dictionary of arguments from the child class's constructor
            **kwargs: additional arguments to store
        """
        config = kwargs.copy()

        # Extract parameters from the local_vars, excluding 'self' and '__class__'
        args_dict = {
            key: val for key, val in args.items() if key not in ["self", "__class__"]
        }

        config.update(args_dict)

        self.config = config

    def init_model_weights(
        self, kernel: str, bias: str, seed: int | None = None, **kwargs
    ):
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
