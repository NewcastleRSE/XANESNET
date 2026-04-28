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

"""Memory-efficient down-projection, Hadamard, and bilinear interaction layers for GemNet."""

import torch

from ..utils.initializer import he_orthogonal_init


class EfficientInteractionDownProjection(torch.nn.Module):
    """Down-projection used in the efficient reformulation of the basis interaction.

    Projects the radial basis component of the interaction from ``num_radial``
    dimensions to ``emb_size_interm`` using a learned weight tensor.

    Args:
        num_spherical: Number of spherical harmonics (controls the first tensor
            dimension of the weight).
        num_radial: Number of radial basis functions.
        emb_size_interm: Intermediate (down-projected) embedding size.
    """

    def __init__(
        self,
        num_spherical: int,
        num_radial: int,
        emb_size_interm: int,
    ) -> None:
        """Initialize ``EfficientInteractionDownProjection``."""
        super().__init__()

        self.num_spherical = num_spherical
        self.num_radial = num_radial
        self.emb_size_interm = emb_size_interm

        self.weight = torch.nn.Parameter(
            torch.empty((self.num_spherical, self.num_radial, self.emb_size_interm)),
            requires_grad=True,
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Re-initialize the weight tensor with He-orthogonal init."""
        he_orthogonal_init(self.weight)

    def forward(self, tbf: tuple[torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply the down-projection to the radial basis component.

        Args:
            tbf: Tuple ``(rbf_env, sph)`` where

                * ``rbf_env``: shape ``(num_spherical, nEdges, num_radial)``.
                * ``sph``: shape ``(nEdges, Kmax, num_spherical)``.

        Returns:
            Tuple ``(rbf_W1, sph)`` where

            * ``rbf_W1``: shape ``(nEdges, emb_size_interm, num_spherical)``.
            * ``sph``: shape ``(nEdges, num_spherical, Kmax)``.
        """
        rbf_env, sph = tbf
        # rbf_env: (num_spherical, nEdges, num_radial); sph: (nEdges, Kmax, num_spherical)
        # MatMul: multiply and sum over num_radial.
        rbf_W1 = torch.matmul(rbf_env, self.weight)  # (num_spherical, nEdges , emb_size_interm)
        rbf_W1 = rbf_W1.permute(1, 2, 0)  # (nEdges, emb_size_interm, num_spherical)

        sph = torch.transpose(sph, 1, 2)  # (nEdges, num_spherical, Kmax)
        return rbf_W1, sph


class EfficientInteractionHadamard(torch.nn.Module):
    """Efficient Hadamard product and summation over neighbor edge embeddings.

    Args:
        emb_size_interm: Intermediate (down-projected) embedding size.
        emb_size: Edge embedding size.
    """

    def __init__(
        self,
        emb_size_interm: int,
        emb_size: int,
    ) -> None:
        """Initialize ``EfficientInteractionHadamard``."""
        super().__init__()
        self.emb_size_interm = emb_size_interm
        self.emb_size = emb_size

        self.weight = torch.nn.Parameter(torch.empty((self.emb_size, 1, self.emb_size_interm), requires_grad=True))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Re-initialize the weight tensor with He-orthogonal init."""
        he_orthogonal_init(self.weight)

    def forward(
        self,
        basis: tuple[torch.Tensor, torch.Tensor],
        m: torch.Tensor,
        id_reduce: torch.Tensor,
        Kidx: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the Hadamard-based edge message aggregation.

        Args:
            basis: Tuple ``(rbf_W1, sph)`` from
                :class:`EfficientInteractionDownProjection`:

                * ``rbf_W1``: shape ``(nEdges, emb_size_interm, num_spherical)``.
                * ``sph``: shape ``(nEdges, num_spherical, Kmax)``.

            m: Neighbor edge embeddings (``m_db`` for quadruplets, ``m_ba``
                for triplets), shape ``(nTriplets_or_Quadruplets, emb_size)``.
            id_reduce: Index mapping each triplet/quadruplet to its target edge,
                shape ``(nTriplets_or_Quadruplets,)``.
            Kidx: Neighbor position within the zero-padded dense matrix,
                shape ``(nTriplets_or_Quadruplets,)``.

        Returns:
            Aggregated edge embeddings of shape ``(nEdges, emb_size)``.
        """
        # quadruplets: m = m_db; triplets: m = m_ba
        # num_spherical is num_spherical**2 for quadruplets
        rbf_W1, sph = basis  # (nEdges, emb_size_interm, num_spherical), (nEdges, num_spherical, Kmax)
        nEdges = rbf_W1.shape[0]

        # Create zero-padded dense matrix of neighboring edge embeddings.
        # Kmax is already encoded in the basis tensor layout.
        Kmax = sph.shape[2]
        m2 = torch.zeros(nEdges, Kmax, self.emb_size, device=self.weight.device, dtype=m.dtype)
        m2[id_reduce, Kidx] = m  # (nTriplets_or_Quadruplets, emb_size) -> (nEdges, Kmax, emb_size)

        sum_k = torch.matmul(sph, m2)  # (nEdges, num_spherical, emb_size)

        # MatMul: multiply and sum over num_spherical.
        rbf_W1_sum_k = torch.matmul(rbf_W1, sum_k)  # (nEdges, emb_size_interm, emb_size)

        # MatMul: multiply and sum over emb_size_interm.
        m_ca = torch.matmul(self.weight, rbf_W1_sum_k.permute(2, 1, 0))[:, 0]  # (emb_size, nEdges)
        m_ca = torch.transpose(m_ca, 0, 1)  # (nEdges, emb_size)

        return m_ca


class EfficientInteractionBilinear(torch.nn.Module):
    """Efficient bilinear layer and subsequent summation over neighbor edge embeddings.

    Args:
        emb_size: Edge embedding size.
        emb_size_interm: Intermediate (down-projected) embedding size.
        units_out: Output embedding size of the bilinear layer.
    """

    def __init__(
        self,
        emb_size: int,
        emb_size_interm: int,
        units_out: int,
    ) -> None:
        """Initialize ``EfficientInteractionBilinear``."""
        super().__init__()
        self.emb_size = emb_size
        self.emb_size_interm = emb_size_interm
        self.units_out = units_out

        self.weight = torch.nn.Parameter(
            torch.empty(
                (self.emb_size, self.emb_size_interm, self.units_out),
                requires_grad=True,
            )
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Re-initialize the weight tensor with He-orthogonal init."""
        he_orthogonal_init(self.weight)

    def forward(
        self,
        basis: tuple[torch.Tensor, torch.Tensor],
        m: torch.Tensor,
        id_reduce: torch.Tensor,
        Kidx: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the bilinear edge message aggregation.

        Args:
            basis: Tuple ``(rbf_W1, sph)`` from
                :class:`EfficientInteractionDownProjection`:

                * ``rbf_W1``: shape ``(nEdges, emb_size_interm, num_spherical)``.
                * ``sph``: shape ``(nEdges, num_spherical, Kmax)``.

            m: Neighbor edge embeddings, shape
                ``(nTriplets_or_Quadruplets, emb_size)``.
            id_reduce: Index mapping each triplet/quadruplet to its target edge,
                shape ``(nTriplets_or_Quadruplets,)``.
            Kidx: Neighbor position within the zero-padded dense matrix,
                shape ``(nTriplets_or_Quadruplets,)``.

        Returns:
            Aggregated edge embeddings of shape ``(nEdges, units_out)``.
        """
        # quadruplets: m = m_db; triplets: m = m_ba
        # num_spherical is num_spherical**2 for quadruplets
        rbf_W1, sph = basis  # (nEdges, emb_size_interm, num_spherical), (nEdges, num_spherical, Kmax)
        nEdges = rbf_W1.shape[0]

        # Create zero-padded dense matrix of neighboring edge embeddings.
        # Kmax is already encoded in the basis tensor layout.
        Kmax = sph.shape[2]
        m2 = torch.zeros(nEdges, Kmax, self.emb_size, device=self.weight.device, dtype=m.dtype)
        m2[id_reduce, Kidx] = m  # (nTriplets_or_Quadruplets, emb_size) -> (nEdges, Kmax, emb_size)

        sum_k = torch.matmul(sph, m2)  # (nEdges, num_spherical, emb_size)

        # MatMul: multiply and sum over num_spherical.
        rbf_W1_sum_k = torch.matmul(rbf_W1, sum_k)  # (nEdges, emb_size_interm, emb_size)

        # Bilinear: sum over emb_size_interm and emb_size.
        m_ca = torch.matmul(rbf_W1_sum_k.permute(2, 0, 1), self.weight)  # (emb_size, nEdges, units_out)
        m_ca = torch.sum(m_ca, dim=0)  # (nEdges, units_out)
        return m_ca
