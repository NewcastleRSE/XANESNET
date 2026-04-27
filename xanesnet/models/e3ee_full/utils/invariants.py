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

"""Utilities for extracting rotation-invariant features from e3nn irreps tensors."""

import torch
from e3nn import o3


def invariant_feature_dim(irreps: o3.Irreps) -> int:
    """Count the total number of invariant scalar channels in an irreps specification.

    For every ``(mul, ir)`` pair the multiplicity ``mul`` contributes one invariant
    per irrep copy (the l=0 scalar channel directly, or the RMS norm for l>0).

    Args:
        irreps: e3nn irreps specification to inspect.

    Returns:
        Total number of invariant scalar dimensions.
    """
    return sum(mul for mul, _ in irreps)


def invariant_features_from_irreps(x: torch.Tensor, irreps: o3.Irreps) -> torch.Tensor:
    """Convert flattened irreps features to rotation-invariant scalars.

    For ``l = 0`` blocks the scalar channels are returned directly.
    For ``l > 0`` blocks the RMS norm of each irrep copy is returned.

    Args:
        x: Concatenated irreps features of shape ``(..., D)`` where ``D == irreps.dim``.
        irreps: e3nn irreps specification matching the last dimension of ``x``.

    Returns:
        Invariant features of shape ``(..., inv_dim)``
        where ``inv_dim == invariant_feature_dim(irreps)``.
    """
    orig_shape = x.shape[:-1]
    d = x.shape[-1]
    x_flat = x.reshape(-1, d)
    m = x_flat.shape[0]

    outs: list[torch.Tensor] = []
    offset = 0
    for mul, ir in irreps:
        dim = ir.dim
        block_dim = mul * dim
        xb = x_flat[:, offset : offset + block_dim].reshape(m, mul, dim)

        if ir.l == 0:
            outs.append(xb.reshape(m, mul))
        else:
            inv = torch.sqrt((xb**2).mean(dim=-1) + 1e-8)
            outs.append(inv)

        offset += block_dim

    out = torch.cat(outs, dim=-1)
    return out.view(*orig_shape, out.shape[-1])
