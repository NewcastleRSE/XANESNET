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

"""He-orthogonal weight initialisation for GemNet layers."""

import math

import torch


def _standardize(kernel: torch.Tensor) -> torch.Tensor:
    """Standardise a weight tensor to zero mean and unit variance.

    For 3-D tensors (e.g. the weight in
    :class:`~xanesnet.models.gemnet.layers.efficient.EfficientInteractionDownProjection`),
    statistics are computed over the first two axes; for 2-D tensors over axis 1.

    Args:
        kernel: Weight tensor of shape ``(*, out_features)`` or
            ``(in_a, in_b, out_features)``.

    Returns:
        Standardised tensor with the same shape as ``kernel``.
    """
    eps = 1e-6

    axis: int | list[int]
    if len(kernel.shape) == 3:
        axis = [0, 1]  # last dimension is output dimension
    else:
        axis = 1

    var, mean = torch.var_mean(kernel, dim=axis, unbiased=True, keepdim=True)
    kernel = (kernel - mean) / (var + eps) ** 0.5
    return kernel


def he_orthogonal_init(tensor: torch.Tensor) -> torch.Tensor:
    """Initialise a weight tensor with He-variance using a random orthogonal matrix.

    Applies orthogonal initialisation and then rescales to achieve variance
    ``1 / fan_in``, following He et al. ("Delving deep into rectifiers").
    Using a (semi-)orthogonal initialisation decorrelates features, which has
    been found to improve training.

    Args:
        tensor: Weight tensor to initialise in-place. Supported shapes:
            ``(out_features, in_features)`` or
            ``(dim_a, dim_b, out_features)`` for the efficient-interaction weights.

    Returns:
        The initialised ``tensor`` (same object, modified in-place).
    """
    tensor = torch.nn.init.orthogonal_(tensor)

    if len(tensor.shape) == 3:
        fan_in = math.prod(tensor.shape[:-1])
    else:
        fan_in = tensor.shape[1]

    with torch.no_grad():
        tensor.data = _standardize(tensor.data)
        tensor.data *= (1 / fan_in) ** 0.5

    return tensor
