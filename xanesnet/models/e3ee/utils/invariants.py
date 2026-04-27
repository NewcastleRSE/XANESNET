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
    """Return the number of invariant scalar channels extractable from an irreps.

    Each irrep copy contributes one invariant channel regardless of its ``l``
    order (scalars are kept directly; higher-order irreps are reduced to their
    RMS norm). The result equals the total multiplicity summed over all irreps.

    Args:
        irreps: The e3nn irreps specification.

    Returns:
        Total multiplicity (number of invariant channels).
    """
    return sum(mul for mul, _ in irreps)


def invariant_features_from_irreps(x: torch.Tensor, irreps: o3.Irreps) -> torch.Tensor:
    """Convert flattened irreps features to rotation-invariant features.

    For ``l = 0`` irreps the scalar channels are kept directly. For ``l > 0``
    irreps each copy is replaced by its RMS norm.

    Args:
        x: Concatenated irreps features of shape ``(..., D)`` where ``D``
            equals ``irreps.dim``.
        irreps: The e3nn irreps specification matching the last dimension of
            ``x``.

    Returns:
        Invariant features of shape ``(..., inv_dim)`` where ``inv_dim`` equals
        :func:`invariant_feature_dim` applied to ``irreps``.
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
