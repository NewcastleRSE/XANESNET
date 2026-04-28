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

"""Environment Embedding (EnvEmbed) prediction model."""

import torch
import torch.nn as nn

from xanesnet.components import BiasInitRegistry, WeightInitRegistry
from xanesnet.serialization.config import Config
from xanesnet.utils.math import SpectralBasis, gaussian_inverse

from ..base import Model
from ..registry import ModelRegistry
from .layers import (
    CoeffHeadGroupedResidualPreLN,
    ResidualPreLNBlock,
    SoftRadialShellsEncoder,
)


@ModelRegistry.register("envembed")
class EnvEmbed(Model):
    """Environment Embedding model.

    Architecture:

    1. :class:`~xanesnet.models.envembed.layers.SoftRadialShellsEncoder`: learnable
       soft radial shell binning over absorber-centric distances, fused with the
       absorber descriptor to produce a fixed-size latent vector.
    2. :class:`~xanesnet.models.envembed.layers.CoeffHeadGroupedResidualPreLN`: shared
       Pre-LN residual trunk predicting spectral basis coefficients per width group.

    Args:
        model_type: Model type string (passed to base class).
        in_size: Descriptor feature dimension ``H``.
        kgroups: Number of spectral basis coefficients per width group.
        n_shells: Number of learnable radial shells.
        max_radius_angs: Radial cutoff in **A**.
        init_width: Initial Gaussian shell width in **A**.
        use_gating: If ``True``, modulate shell summary with Fourier distance features.
        head_hidden: Hidden dimension of the residual blocks in the coefficient head.
        head_depth: Number of :class:`~xanesnet.models.envembed.layers.ResidualPreLNBlock`
            layers in the coefficient head trunk.
        dropout: Dropout probability applied in the coefficient head residual blocks.
    """

    def __init__(
        self,
        model_type: str,
        # params:
        in_size: int,
        kgroups: list[int],
        n_shells: int,
        max_radius_angs: float,
        init_width: float,
        use_gating: bool,
        head_hidden: int,
        head_depth: int,
        dropout: float,
    ) -> None:
        """Initialize ``EnvEmbed``."""
        super().__init__(model_type)

        self.in_size = in_size
        self.kgroups = kgroups
        self.n_shells = n_shells
        self.max_radius_angs = max_radius_angs
        self.init_width = init_width
        self.use_gating = use_gating
        self.head_hidden = head_hidden
        self.head_depth = head_depth
        self.dropout = dropout

        latent_dim = in_size * 2

        self.encoder = SoftRadialShellsEncoder(
            d_input=in_size,
            n_shells=n_shells,
            latent_dim=latent_dim,
            max_radius_angs=max_radius_angs,
            init_width=init_width,
            use_gating=use_gating,
        )

        self.coeff_head = CoeffHeadGroupedResidualPreLN(
            latent_dim=latent_dim,
            k_groups=kgroups,
            hidden=head_hidden,
            depth=head_depth,
            dropout=dropout,
        )

    def forward(
        self,
        descriptor_features: torch.Tensor,
        distance_features: torch.Tensor,
        lengths: torch.Tensor,
        basis: SpectralBasis,
    ) -> torch.Tensor:
        """Encode the local chemical environment and predict the XANES spectrum.

        Args:
            descriptor_features: Padded descriptor features with absorber at index 0, shape ``(B, N, H)``.
            distance_features: Distances from the absorber atom, shape ``(B, N)``.
            lengths: Number of real atoms per sample (before padding), shape ``(B,)``.
            basis: Spectral basis used to reconstruct the spectrum from coefficients.

        Returns:
            Predicted XANES spectra of shape ``(B, n_energies)``.
        """
        h = self.encoder(descriptor_features, dists=distance_features, lengths=lengths)
        coeff = self.coeff_head(h)
        # TODO can we do better if directly returning coefficients?
        return gaussian_inverse(basis=basis, coeffs=coeff)

    def init_weights(self, weights_init: str, bias_init: str, **kwargs) -> None:
        """Initialize model weights and biases.

        Applies the given initialization to ``nn.Linear`` layers in the encoder.
        For the coefficient head, :class:`~xanesnet.models.envembed.layers.ResidualPreLNBlock`
        inner layers are initialized while group head layers retain their zero initialization.

        Args:
            weights_init: Name of the weight initializer registered in
                :class:`~xanesnet.components.WeightInitRegistry`.
            bias_init: Name of the bias initializer registered in
                :class:`~xanesnet.components.BiasInitRegistry`.
            **kwargs: Additional keyword arguments forwarded to the weight initializer.
        """
        weight_init_fn = WeightInitRegistry.get(weights_init, **kwargs)
        bias_init_fn = BiasInitRegistry.get(bias_init)

        for module in self.encoder.modules():
            if isinstance(module, nn.Linear):
                weight_init_fn(module.weight)
                if module.bias is not None:
                    bias_init_fn(module.bias)

        for module in self.coeff_head.modules():
            if isinstance(module, ResidualPreLNBlock):
                weight_init_fn(module.fc1.weight)
                bias_init_fn(module.fc1.bias)
                weight_init_fn(module.fc2.weight)
                bias_init_fn(module.fc2.bias)

    @property
    def signature(self) -> Config:
        """Return the model signature as a :class:`~xanesnet.serialization.config.Config`."""
        signature = super().signature
        signature.update_with_dict(
            {
                "in_size": self.in_size,
                "kgroups": self.kgroups,
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
