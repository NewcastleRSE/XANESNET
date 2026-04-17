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

from typing import Any

import torch

from xanesnet.serialization.config import Config

from ..base import Model
from ..registry import ModelRegistry


@ModelRegistry.register("envembed")
class EnvEmbed(Model):
    def __init__(
        self,
        model_type: str,
        # params:
        in_size: Any,  # TODO specify type more precisely
        kgroups: Any,  # TODO specify type more precisely
        out_size: int,
        n_shells: int,
        max_radius_angs: float,
        init_width: float,
        use_gating: bool,
        head_hidden: int,
        head_depth: int,
        dropout: float,
    ) -> None:
        super().__init__(model_type)

        self.in_size = in_size
        self.kgroups = kgroups
        self.out_size = out_size
        self.n_shells = n_shells
        self.max_radius_angs = max_radius_angs
        self.init_width = init_width
        self.use_gating = use_gating
        self.head_hidden = head_hidden
        self.head_depth = head_depth
        self.dropout = dropout

        # TODO implement init

        # Placeholder linear layer such that parameters are not empty
        # TODO should be removed
        linear_layer = torch.nn.Linear(in_size, out_size)
        self.add_module("linear_layer", linear_layer)

    # TODO implement rest
    # TODO put layers in layers folder for clean structure similar to E3EE

    def init_weights(self, weights_init: str, bias_init: str, **kwargs) -> None:
        # TODO implement weight initialization
        pass

    @property
    def signature(self) -> Config:
        """
        Return model signature as a configuration dictionary.
        """
        signature = super().signature
        signature.update_with_dict(
            {
                "in_size": self.in_size,
                "kgroups": self.kgroups,
                "out_size": self.out_size,
                "n_shells": self.n_shells,
                "max_radius_angs": self.max_radius_angs,
                "init_width": self.init_width,
                "use_gating": self.use_gating,
                "head_hidden": self.head_hidden,
                "head_depth": self.head_depth,
                "dropout": self.dropout,
            }
        )
        return signature
        return signature
