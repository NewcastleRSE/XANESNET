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

"""Efficient basis embedding and bilinear interaction layers for GemNet-OC."""

import torch

from ..utils import he_orthogonal_init
from .base_layers import Dense


class BasisEmbedding(torch.nn.Module):
    """Project a radial and optional spherical basis through a weight matrix.

    Supports the efficient reformulation where intermediate triplet/quadruplet
    bases are zero-padded into dense tensors for batch matrix operations.

    Args:
        num_radial: Number of radial basis functions.
        emb_size_interm: Intermediate embedding dimension.
        num_spherical: Number of circular/spherical basis functions. Required
            only when a circular or spherical basis is present.
    """

    weight: torch.nn.Parameter

    def __init__(
        self,
        num_radial: int,
        emb_size_interm: int,
        num_spherical: int | None = None,
    ) -> None:
        super().__init__()
        self.num_radial = num_radial
        self.num_spherical = num_spherical
        if num_spherical is None:
            self.weight = torch.nn.Parameter(
                torch.empty(emb_size_interm, num_radial),
                requires_grad=True,
            )
        else:
            self.weight = torch.nn.Parameter(
                torch.empty(num_radial, num_spherical, emb_size_interm),
                requires_grad=True,
            )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Re-initialise weights with He-orthogonal initialisation."""
        he_orthogonal_init(self.weight)

    def forward(
        self,
        rad_basis: torch.Tensor,
        sph_basis: torch.Tensor | None = None,
        idx_rad_outer: torch.Tensor | None = None,
        idx_rad_inner: torch.Tensor | None = None,
        idx_sph_outer: torch.Tensor | None = None,
        idx_sph_inner: torch.Tensor | None = None,
        num_atoms: int | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Project the radial and optional spherical basis.

        Args:
            rad_basis: Raw radial basis, shape
                ``(num_edges, num_radial)`` or
                ``(num_edges, num_orders * num_radial)``.
            sph_basis: Raw circular or spherical basis, shape
                ``(num_triplets_or_quadruplets, num_spherical)``.
            idx_rad_outer: Atom index for each edge, shape ``(num_edges,)``.
                Used for efficient edge aggregation.
            idx_rad_inner: Per-atom edge enumeration, shape ``(num_edges,)``.
                Used for efficient edge aggregation.
            idx_sph_outer: Edge index for each triplet/quadruplet,
                shape ``(num_triplets_or_quadruplets,)``.
            idx_sph_inner: Per-edge triplet enumeration,
                shape ``(num_triplets_or_quadruplets,)``.
            num_atoms: Total number of atoms; required when
                ``idx_rad_inner`` is provided.

        Returns:
            When only a radial basis is present without dense packing:
            ``rad_W1`` of shape ``(num_edges, emb_size_interm)``.
            When only a radial basis is present with dense packing:
            ``rad_W1`` of shape ``(num_atoms, emb_size_interm, Kmax)``.
            When a spherical basis is present without dense packing:
            ``sph_W1`` of shape
            ``(num_triplets_or_quadruplets, emb_size_interm)``.
            When dense packing is requested: a tuple
            ``(rad_W1, sph2)`` with shapes
            ``(num_edges_or_atoms, emb_size_interm, Kmax2 * num_spherical)`` and
            ``(num_edges, num_spherical, Kmax)``.
        """
        num_edges = rad_basis.shape[0]

        if self.num_spherical is not None:
            assert sph_basis is not None
            # MatMul: mul + sum over num_radial
            rad_W1 = rad_basis @ self.weight.reshape(self.weight.shape[0], -1)
            # (num_edges, emb_size_interm * num_spherical)
            rad_W1 = rad_W1.reshape(num_edges, -1, sph_basis.shape[-1])
            # (num_edges, emb_size_interm, num_spherical)
        else:
            # MatMul: mul + sum over num_radial
            rad_W1 = rad_basis @ self.weight.T
            # (num_edges, emb_size_interm)

        if idx_rad_inner is not None:
            assert idx_rad_outer is not None
            assert num_atoms is not None
            # Zero padded dense matrix
            # maximum number of neighbors

            Kmax = 0 if idx_rad_outer.shape[0] == 0 else int(torch.max(idx_rad_inner).item()) + 1

            rad_W1_padded = rad_W1.new_zeros([num_atoms, Kmax, *list(rad_W1.shape[1:])])
            rad_W1_padded[idx_rad_outer, idx_rad_inner] = rad_W1
            # (num_atoms, Kmax, emb_size_interm, ...)
            rad_W1_padded = torch.transpose(rad_W1_padded, 1, 2)
            # (num_atoms, emb_size_interm, Kmax, ...)
            rad_W1_padded = rad_W1_padded.reshape(num_atoms, int(rad_W1.shape[1]), -1)
            # (num_atoms, emb_size_interm, Kmax2 * ...)
            rad_W1 = rad_W1_padded

        sph2: torch.Tensor | None = None
        if idx_sph_inner is not None:
            assert idx_sph_outer is not None
            assert sph_basis is not None
            # Zero padded dense matrix
            # maximum number of neighbors
            Kmax = 0 if idx_sph_outer.shape[0] == 0 else int(torch.max(idx_sph_inner).item()) + 1

            sph2_local: torch.Tensor = sph_basis.new_zeros(num_edges, Kmax, sph_basis.shape[-1])
            sph2_local[idx_sph_outer, idx_sph_inner] = sph_basis
            # (num_edges, Kmax, num_spherical)
            sph2 = torch.transpose(sph2_local, 1, 2)
            # (num_edges, num_spherical, Kmax)

        if sph_basis is None:
            return rad_W1
        else:
            if idx_sph_inner is None:
                assert idx_sph_outer is not None
                rad_W1 = rad_W1[idx_sph_outer]
                # (num_triplets, emb_size_interm, num_spherical)

                sph_W1 = rad_W1 @ sph_basis[:, :, None]
                # (num_triplets, emb_size_interm, num_spherical)
                return sph_W1.squeeze(-1)
            else:
                assert sph2 is not None
                return rad_W1, sph2


class EfficientInteractionBilinear(torch.nn.Module):
    """Efficient bilinear layer that contracts basis tensors against edge embeddings.

    Reformulates the standard bilinear contraction into batch matrix
    multiplications over zero-padded dense tensors for GPU efficiency.

    Args:
        emb_size_in: Input triplet/quadruplet embedding dimension.
        emb_size_interm: Intermediate (basis) embedding dimension.
        emb_size_out: Output triplet/quadruplet embedding dimension.
    """

    def __init__(
        self,
        emb_size_in: int,
        emb_size_interm: int,
        emb_size_out: int,
    ) -> None:
        super().__init__()
        self.emb_size_in = emb_size_in
        self.emb_size_interm = emb_size_interm
        self.emb_size_out = emb_size_out

        self.bilinear = Dense(
            self.emb_size_in * self.emb_size_interm,
            self.emb_size_out,
            bias=False,
            activation=None,
        )

    def forward(
        self,
        basis: tuple[torch.Tensor, torch.Tensor],
        m: torch.Tensor,
        idx_agg_outer: torch.Tensor,
        idx_agg_inner: torch.Tensor,
        idx_agg2_outer: torch.Tensor | None = None,
        idx_agg2_inner: torch.Tensor | None = None,
        agg2_out_size: int | None = None,
    ) -> torch.Tensor:
        """Aggregate edge messages via a bilinear basis contraction.

        Args:
            basis: Tuple ``(rad_W1, sph)`` where ``rad_W1`` has shape
                ``(num_edges, emb_size_interm, num_spherical)`` for a single
                aggregation or ``(num_atoms, emb_size_interm, Kmax2 *
                num_spherical)`` for a double aggregation, while ``sph`` has
                shape ``(num_edges, num_spherical, Kmax)``.
            m: Input triplet/quadruplet embeddings, shape
                ``(num_triplets_or_quadruplets, emb_size_in)``.
            idx_agg_outer: Output edge for each triplet/quadruplet,
                shape ``(num_triplets_or_quadruplets,)``.
            idx_agg_inner: Per-output-edge triplet enumeration,
                shape ``(num_triplets_or_quadruplets,)``.
            idx_agg2_outer: Output atom for each edge, shape
                ``(num_edges,)``. Required for double aggregation.
            idx_agg2_inner: Per-atom edge enumeration, shape
                ``(num_edges,)``. Required for double aggregation.
            agg2_out_size: Number of output embeddings when aggregating
                twice (typically the number of atoms).

        Returns:
            Aggregated edge or atom embeddings of shape
            ``(num_edges_or_atoms, emb_size_out)``.
        """
        # num_spherical is actually num_spherical**2 for quadruplets
        (rad_W1, sph) = basis
        # (num_edges, emb_size_interm, num_spherical),
        # (num_edges, num_spherical, Kmax)
        num_edges = sph.shape[0]

        # Create (zero-padded) dense matrix of the neighboring edge embeddings.
        Kmax = 0 if idx_agg_inner.numel() == 0 else int(idx_agg_inner.max().item()) + 1
        m_padded = m.new_zeros(num_edges, Kmax, self.emb_size_in)
        m_padded[idx_agg_outer, idx_agg_inner] = m
        # (num_quadruplets/num_triplets, emb_size_in) -> (num_edges, Kmax, emb_size_in)

        sph_m = torch.matmul(sph, m_padded)
        # (num_edges, num_spherical, emb_size_in)

        if idx_agg2_outer is not None:
            assert idx_agg2_inner is not None
            assert agg2_out_size is not None
            Kmax2 = 0 if idx_agg2_inner.numel() == 0 else int(idx_agg2_inner.max().item()) + 1
            sph_m_padded = sph_m.new_zeros(agg2_out_size, Kmax2, sph_m.shape[1], sph_m.shape[2])
            sph_m_padded[idx_agg2_outer, idx_agg2_inner] = sph_m
            # (num_atoms, Kmax2, num_spherical, emb_size_in)
            sph_m_padded = sph_m_padded.reshape(agg2_out_size, -1, sph_m.shape[-1])
            # (num_atoms, Kmax2 * num_spherical, emb_size_in)

            rad_W1_sph_m = rad_W1 @ sph_m_padded
            # (num_atoms, emb_size_interm, emb_size_in)
        else:
            # MatMul: mul + sum over num_spherical
            rad_W1_sph_m = torch.matmul(rad_W1, sph_m)
            # (num_edges, emb_size_interm, emb_size_in)

        # Bilinear: Sum over emb_size_interm and emb_size_in
        return self.bilinear(rad_W1_sph_m.reshape(-1, rad_W1_sph_m.shape[1:].numel()))
        # (num_edges/num_atoms, emb_size_out)
