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

from torch import nn

# TODO move the ActivationSwitch to a registry!


class ActivationSwitch:
    """
    A factory class to get activation function instances from their names.
    """

    ACTIVATIONS = {
        "relu": nn.ReLU,
        "prelu": nn.PReLU,
        "tanh": nn.Tanh,
        "sigmoid": nn.Sigmoid,
        "elu": nn.ELU,
        "leakyrelu": nn.LeakyReLU,
        "selu": nn.SELU,
        "silu": nn.SiLU,
        "gelu": nn.GELU,
    }

    def get(self, activation_name: str, **kwargs) -> nn.Module:
        activation_name_lower = activation_name.lower()
        if activation_name_lower not in self.ACTIVATIONS:
            raise TypeError(f"Invalided activation function name '{activation_name}'.")

        activation_class = self.ACTIVATIONS[activation_name_lower]
        return activation_class(**kwargs)
