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

import torch

from ..utils.initializer import he_orthogonal_init


class EfficientInteractionDownProjection(torch.nn.Module):
    """
    Down projection in the efficient reformulation.

    Parameters
    ----------
        num_spherical: int
            Same as the setting in the basis layers.
        num_radial: int
            Same as the setting in the basis layers.
        emb_size_interm: int
            Intermediate embedding size (down-projection size).
    """

    def __init__(
        self,
        num_spherical: int,
        num_radial: int,
        emb_size_interm: int,
    ) -> None:
        super().__init__()

        self.num_spherical = num_spherical
        self.num_radial = num_radial
        self.emb_size_interm = emb_size_interm

        self.weight = torch.nn.Parameter(
            torch.empty((self.num_spherical, self.num_radial, self.emb_size_interm)),
            requires_grad=True,
        )

    def reset_parameters(self) -> None:
        he_orthogonal_init(self.weight)

    def forward(self, tbf: tuple[torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
            (rbf_W1, sph): tuple
            - rbf_W1: Tensor, shape=(nEdges, emb_size_interm, num_spherical)
            - sph: Tensor, shape=(nEdges, Kmax, num_spherical)
        """
        rbf_env, sph = tbf
        # (num_spherical, nEdges, num_radial), (nEdges, Kmax, num_spherical) ;  Kmax = maximum number of neighbors of the edges

        # MatMul: mul + sum over num_radial
        rbf_W1 = torch.matmul(rbf_env, self.weight)  # (num_spherical, nEdges , emb_size_interm)
        rbf_W1 = rbf_W1.permute(1, 2, 0)  # (nEdges, emb_size_interm, num_spherical)

        sph = torch.transpose(sph, 1, 2)  # (nEdges, num_spherical, Kmax)
        return rbf_W1, sph


class EfficientInteractionHadamard(torch.nn.Module):
    """
    Efficient reformulation of the hadamard product and subsequent summation.

    Parameters
    ----------
        emb_size_interm: int
            Intermediate embedding size (down-projection size).
        emb_size: int
            Embedding size.
    """

    def __init__(
        self,
        emb_size_interm: int,
        emb_size: int,
    ) -> None:
        super().__init__()
        self.emb_size_interm = emb_size_interm
        self.emb_size = emb_size

        self.weight = torch.nn.Parameter(torch.empty((self.emb_size, 1, self.emb_size_interm), requires_grad=True))

    def reset_parameters(self) -> None:
        he_orthogonal_init(self.weight)

    def forward(
        self,
        basis: tuple[torch.Tensor, torch.Tensor],
        m: torch.Tensor,
        id_reduce: torch.Tensor,
        Kidx: torch.Tensor,
    ) -> torch.Tensor:
        """
        Returns
        -------
            m_ca: Tensor, shape=(nEdges, emb_size)
                Edge embeddings.
        """
        # quadruplets: m = m_db , triplets: m = m_ba
        # num_spherical is actually num_spherical**2 for quadruplets
        rbf_W1, sph = basis  # (nEdges, emb_size_interm, num_spherical) ,  (nEdges, num_spherical, Kmax)
        nEdges = rbf_W1.shape[0]

        # Create (zero-padded) dense matrix of the neighboring edge embeddings.
        # maximum number of neighbors, catch empty id_reduce_ji with maximum
        if sph.shape[2] == 0:
            Kmax = 0
        else:
            Kmax = int(torch.max(torch.max(Kidx + 1), torch.tensor(0)).item())
        m2 = torch.zeros(nEdges, Kmax, self.emb_size, device=self.weight.device, dtype=m.dtype)
        m2[id_reduce, Kidx] = m  # (nQuadruplets or nTriplets, emb_size) -> (nEdges, Kmax, emb_size)

        sum_k = torch.matmul(sph, m2)  # (nEdges, num_spherical, emb_size)

        # MatMul: mul + sum over num_spherical
        rbf_W1_sum_k = torch.matmul(rbf_W1, sum_k)  # (nEdges, emb_size_interm, emb_size)

        # MatMul: mul + sum over emb_size_interm
        m_ca = torch.matmul(self.weight, rbf_W1_sum_k.permute(2, 1, 0))[:, 0]  # (emb_size, nEdges)
        m_ca = torch.transpose(m_ca, 0, 1)  # (nEdges, emb_size)

        return m_ca


class EfficientInteractionBilinear(torch.nn.Module):
    """
    Efficient reformulation of the bilinear layer and subsequent summation.

    Parameters
    ----------
        emb_size: int
            Edge embedding size.
        emb_size_interm: int
            Intermediate embedding size (down-projection size).
        units_out: int
            Embedding output size of the bilinear layer.
        kernel_initializer: callable
            Initializer of the weight matrix.
    """

    def __init__(
        self,
        emb_size: int,
        emb_size_interm: int,
        units_out: int,
    ) -> None:
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

    def reset_parameters(self) -> None:
        he_orthogonal_init(self.weight)

    def forward(
        self,
        basis: tuple[torch.Tensor, torch.Tensor],
        m: torch.Tensor,
        id_reduce: torch.Tensor,
        Kidx: torch.Tensor,
    ) -> torch.Tensor:
        """
        Returns
        -------
            m_ca: Tensor, shape=(nEdges, units_out)
                Edge embeddings.
        """
        # quadruplets: m = m_db , triplets: m = m_ba
        # num_spherical is actually num_spherical**2 for quadruplets
        rbf_W1, sph = basis  # (nEdges, emb_size_interm, num_spherical) ,  (nEdges, num_spherical, Kmax)
        nEdges = rbf_W1.shape[0]

        # Create (zero-padded) dense matrix of the neighboring edge embeddings.
        # maximum number of neighbors, catch empty id_reduce_ji with maximum
        Kmax = 0 if sph.shape[2] == 0 else int(torch.max(torch.max(Kidx + 1), torch.tensor(0)).item())
        m2 = torch.zeros(nEdges, Kmax, self.emb_size, device=self.weight.device, dtype=m.dtype)
        m2[id_reduce, Kidx] = m  # (nQuadruplets or nTriplets, emb_size) -> (nEdges, Kmax, emb_size)

        sum_k = torch.matmul(sph, m2)  # (nEdges, num_spherical, emb_size)

        # MatMul: mul + sum over num_spherical
        rbf_W1_sum_k = torch.matmul(rbf_W1, sum_k)  # (nEdges, emb_size_interm, emb_size)

        # Bilinear: Sum over emb_size_interm and emb_size
        m_ca = torch.matmul(rbf_W1_sum_k.permute(2, 0, 1), self.weight)  # (emb_size, nEdges, units_out)
        m_ca = torch.sum(m_ca, dim=0)  # (nEdges, units_out)
        return m_ca
