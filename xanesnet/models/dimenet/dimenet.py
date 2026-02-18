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

import functools
import logging
import math
from collections.abc import Callable
from typing import Any

import numpy as np
import numpy.typing as npt
import scipy
import sympy as sp
import torch
from torch_geometric.nn import radius_graph
from torch_geometric.nn.inits import glorot_orthogonal
from torch_geometric.nn.resolver import activation_resolver
from torch_geometric.typing import OptTensor, SparseTensor
from torch_geometric.utils import scatter

from xanesnet.serialization.config import Config

from ..base import Model
from ..registry import ModelRegistry


@ModelRegistry.register("dimenet")
class DimeNet(Model):
    """
    The directional message passing neural network DimeNet:
    `"Directional Message Passing for Molecular Graphs"`;
    Arxiv: `<https://arxiv.org/abs/2003.03123>`;
    Implementation similar to `<https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/nn/models/dimenet.html>`
    """

    def __init__(
        self,
        model_type: str,
        # params:
        hidden_channels: int,
        out_channels: int,
        num_blocks: int,
        num_bilinear: int,
        num_spherical: int,
        num_radial: int,
        cutoff: float,
        max_num_neighbors: int,
        envelope_exponent: int,
        num_before_skip: int,
        num_after_skip: int,
        num_output_layers: int,
        act: str,
        output_initializer: str,
    ) -> None:
        super().__init__(model_type)

        if num_spherical < 2:
            raise ValueError("'num_spherical' should be greater than 1")

        # Get activation function from string
        self.act = activation_resolver(act)

        # params
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_blocks = num_blocks
        self.num_bilinear = num_bilinear
        self.num_spherical = num_spherical
        self.num_radial = num_radial
        self.cutoff = cutoff
        self.max_num_neighbors = max_num_neighbors
        self.envelope_exponent = envelope_exponent
        self.num_before_skip = num_before_skip
        self.num_after_skip = num_after_skip
        self.num_output_layers = num_output_layers
        self.output_initializer = output_initializer

        # Distance encoding with bessel functions
        self.rbf = BesselBasisLayer(num_radial, cutoff, envelope_exponent)

        # Combined distance and angle encoding using bessel functions and spherical harmonics
        self.sbf = SphericalBasisLayer(num_spherical, num_radial, cutoff, envelope_exponent)

        # Embedding
        self.emb = EmbeddingBlock(num_radial, hidden_channels, self.act)

        # Output blocks
        self.output_blocks = torch.nn.ModuleList(
            [
                OutputBlock(
                    num_radial,
                    hidden_channels,
                    out_channels,
                    num_output_layers,
                    self.act,
                    output_initializer,
                )
                for _ in range(num_blocks + 1)
            ]
        )

        # Interaction blocks
        self.interaction_blocks = torch.nn.ModuleList(
            [
                InteractionBlock(
                    hidden_channels,
                    num_bilinear,
                    num_spherical,
                    num_radial,
                    num_before_skip,
                    num_after_skip,
                    self.act,
                )
                for _ in range(num_blocks)
            ]
        )

    def forward(
        self,
        z: torch.Tensor,  # Atomic number of each atom with shape [num_atoms].
        pos: torch.Tensor,  # Coordinates of each atom with shape [num_atoms, 3].
        batch: OptTensor = None,  # Batch indices assigning each atom to a separate molecule with shape [num_atoms].
    ) -> torch.Tensor:
        """
        Forward pass.
        """
        # Create edges based on atom positions to all points within the cutoff distance.
        edge_index = radius_graph(pos, r=self.cutoff, batch=batch, max_num_neighbors=self.max_num_neighbors)

        # Constructing triplets of nodes (k→j→i) based on the edges defined in edge_index.
        # It ensures that k and i are distinct to avoid degenerate cases.
        i, j, idx_i, idx_j, idx_k, idx_kj, idx_ji = self.triplets(edge_index, num_nodes=z.size(0))

        # Calculate distances.
        dist = (pos[i] - pos[j]).pow(2).sum(dim=-1).sqrt()

        # Calculate angles.
        pos_ji, pos_ki = pos[idx_j] - pos[idx_i], pos[idx_k] - pos[idx_i]
        a = (pos_ji * pos_ki).sum(dim=-1)
        b = torch.cross(pos_ji, pos_ki, dim=1).norm(dim=-1)
        angle = torch.atan2(b, a)

        # Encoding
        rbf = self.rbf(dist)
        sbf = self.sbf(dist, angle, idx_kj)

        # Embedding block.
        x = self.emb(z, rbf, i, j)
        P = self.output_blocks[0](x, rbf, i, num_nodes=pos.size(0))

        # Interaction blocks.
        for interaction_block, output_block in zip(self.interaction_blocks, self.output_blocks[1:]):
            x = interaction_block(x, rbf, sbf, idx_kj, idx_ji)
            P = P + output_block(x, rbf, i, num_nodes=pos.size(0))

        if batch is None:
            return P.sum(dim=0)
        else:
            return scatter(P, batch, dim=0, reduce="sum")

    @staticmethod
    def triplets(
        edge_index: torch.Tensor,
        num_nodes: int,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        row, col = edge_index  # j->i

        value = torch.arange(row.size(0), device=row.device)
        adj_t = SparseTensor(row=col, col=row, value=value, sparse_sizes=(num_nodes, num_nodes))
        adj_t_row = adj_t.index_select(0, row)  # type: ignore[attr-defined]  # not in torch_sparse stubs
        num_triplets = adj_t_row.set_value(None).sum(dim=1).to(torch.long)

        # Node indices (k->j->i) for triplets.
        idx_i = col.repeat_interleave(num_triplets)
        idx_j = row.repeat_interleave(num_triplets)
        idx_k = adj_t_row.storage.col()
        mask = idx_i != idx_k  # Remove i == k triplets.
        idx_i, idx_j, idx_k = idx_i[mask], idx_j[mask], idx_k[mask]

        # Edge indices (k-j, j->i) for triplets.
        idx_kj = adj_t_row.storage.value()[mask]
        idx_ji = adj_t_row.storage.row()[mask]

        return col, row, idx_i, idx_j, idx_k, idx_kj, idx_ji

    def init_weights(self, weights_init: str, bias_init: str, **kwargs) -> None:
        logging.warning(
            "DimeNet uses custom weight initialization, so 'weights_init' and 'bias_init' arguments are ignored."
        )
        self.rbf.reset_parameters()
        self.emb.reset_parameters()
        for out in self.output_blocks:
            out.reset_parameters()
        for interaction in self.interaction_blocks:
            interaction.reset_parameters()

    @property
    def signature(self) -> Config:
        """
        Return model signature as a dictionary.
        """
        signature = super().signature
        signature.update_with_dict(
            {
                "hidden_channels": self.hidden_channels,
                "out_channels": self.out_channels,
                "num_blocks": self.num_blocks,
                "num_bilinear": self.num_bilinear,
                "num_spherical": self.num_spherical,
                "num_radial": self.num_radial,
                "cutoff": self.cutoff,
                "max_num_neighbors": self.max_num_neighbors,
                "envelope_exponent": self.envelope_exponent,
                "num_before_skip": self.num_before_skip,
                "num_after_skip": self.num_after_skip,
                "num_output_layers": self.num_output_layers,
                "act": self.act.__name__,
                "output_initializer": self.output_initializer,
            }
        )
        return signature


class EmbeddingBlock(torch.nn.Module):
    # Atomic numbers are represented by learnable, randomly initialized
    # atom type embeddings that are shared across molecules
    def __init__(
        self,
        num_radial: int,
        hidden_channels: int,
        act: Callable,
    ) -> None:
        super().__init__()
        self.act = act

        self.emb = torch.nn.Embedding(95, hidden_channels)
        self.lin_rbf = torch.nn.Linear(num_radial, hidden_channels)
        self.lin = torch.nn.Linear(3 * hidden_channels, hidden_channels)

    def reset_parameters(self) -> None:
        self.emb.weight.data.uniform_(-math.sqrt(3), math.sqrt(3))
        self.lin_rbf.reset_parameters()
        self.lin.reset_parameters()

    def forward(
        self,
        x: torch.Tensor,
        rbf: torch.Tensor,
        i: torch.Tensor,
        j: torch.Tensor,
    ) -> torch.Tensor:
        x = self.emb(x)
        rbf = self.act(self.lin_rbf(rbf))
        return self.act(self.lin(torch.cat([x[i], x[j], rbf], dim=-1)))


class InteractionBlock(torch.nn.Module):

    def __init__(
        self,
        hidden_channels: int,
        num_bilinear: int,
        num_spherical: int,
        num_radial: int,
        num_before_skip: int,
        num_after_skip: int,
        act: Callable,
    ) -> None:
        super().__init__()
        self.act = act

        self.lin_rbf = torch.nn.Linear(num_radial, hidden_channels, bias=False)
        self.lin_sbf = torch.nn.Linear(num_spherical * num_radial, num_bilinear, bias=False)

        # Dense transformations of input messages
        self.lin_kj = torch.nn.Linear(hidden_channels, hidden_channels)
        self.lin_ji = torch.nn.Linear(hidden_channels, hidden_channels)

        self.W = torch.nn.Parameter(torch.empty(hidden_channels, num_bilinear, hidden_channels))

        self.layers_before_skip = torch.nn.ModuleList(
            [ResidualLayer(hidden_channels, act) for _ in range(num_before_skip)]
        )
        self.lin = torch.nn.Linear(hidden_channels, hidden_channels)
        self.layers_after_skip = torch.nn.ModuleList(
            [ResidualLayer(hidden_channels, act) for _ in range(num_after_skip)]
        )

    def reset_parameters(self) -> None:
        glorot_orthogonal(self.lin_rbf.weight, scale=2.0)
        glorot_orthogonal(self.lin_sbf.weight, scale=2.0)
        glorot_orthogonal(self.lin_kj.weight, scale=2.0)
        self.lin_kj.bias.data.fill_(0)
        glorot_orthogonal(self.lin_ji.weight, scale=2.0)
        self.lin_ji.bias.data.fill_(0)
        self.W.data.normal_(mean=0, std=2 / self.W.size(0))
        for res_layer in self.layers_before_skip:
            res_layer.reset_parameters()
        glorot_orthogonal(self.lin.weight, scale=2.0)
        self.lin.bias.data.fill_(0)
        for res_layer in self.layers_after_skip:
            res_layer.reset_parameters()

    def forward(
        self,
        x: torch.Tensor,
        rbf: torch.Tensor,
        sbf: torch.Tensor,
        idx_kj: torch.Tensor,
        idx_ji: torch.Tensor,
    ) -> torch.Tensor:
        rbf = self.lin_rbf(rbf)
        sbf = self.lin_sbf(sbf)

        x_ji = self.act(self.lin_ji(x))
        x_kj = self.act(self.lin_kj(x))
        x_kj = x_kj * rbf
        x_kj = torch.einsum("wj,wl,ijl->wi", sbf, x_kj[idx_kj], self.W)
        x_kj = scatter(x_kj, idx_ji, dim=0, dim_size=x.size(0), reduce="sum")

        h = x_ji + x_kj
        for layer in self.layers_before_skip:
            h = layer(h)
        h = self.act(self.lin(h)) + x
        for layer in self.layers_after_skip:
            h = layer(h)

        return h


class ResidualLayer(torch.nn.Module):

    def __init__(
        self,
        hidden_channels: int,
        act: Callable,
    ) -> None:
        super().__init__()
        self.act = act
        self.lin1 = torch.nn.Linear(hidden_channels, hidden_channels)
        self.lin2 = torch.nn.Linear(hidden_channels, hidden_channels)

    def reset_parameters(self) -> None:
        glorot_orthogonal(self.lin1.weight, scale=2.0)
        self.lin1.bias.data.fill_(0)
        glorot_orthogonal(self.lin2.weight, scale=2.0)
        self.lin2.bias.data.fill_(0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.act(self.lin2(self.act(self.lin1(x))))


class OutputBlock(torch.nn.Module):

    def __init__(
        self,
        num_radial: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int,
        act: Callable,
        output_initializer: str = "zeros",  # 'zeros' or 'glorot_orthogonal'
    ) -> None:
        assert output_initializer in {"zeros", "glorot_orthogonal"}

        super().__init__()

        self.act = act
        self.output_initializer = output_initializer

        self.lin_rbf = torch.nn.Linear(num_radial, hidden_channels, bias=False)
        self.lins = torch.nn.ModuleList()
        for _ in range(num_layers):
            self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
        self.lin = torch.nn.Linear(hidden_channels, out_channels, bias=False)

    def reset_parameters(self) -> None:
        glorot_orthogonal(self.lin_rbf.weight, scale=2.0)
        for lin in self.lins:
            glorot_orthogonal(lin.weight, scale=2.0)
            lin.bias.data.fill_(0)
        if self.output_initializer == "zeros":
            self.lin.weight.data.fill_(0)
        elif self.output_initializer == "glorot_orthogonal":
            glorot_orthogonal(self.lin.weight, scale=2.0)

    def forward(
        self,
        x: torch.Tensor,
        rbf: torch.Tensor,
        i: torch.Tensor,
        num_nodes: int | None = None,
    ) -> torch.Tensor:
        x = self.lin_rbf(rbf) * x
        x = scatter(x, i, dim=0, dim_size=num_nodes, reduce="sum")
        for lin in self.lins:
            x = self.act(lin(x))
        return self.lin(x)


class SphericalBasisLayer(torch.nn.Module):

    def __init__(
        self,
        num_spherical: int,
        num_radial: int,
        cutoff: float = 5.0,
        envelope_exponent: int = 5,
    ) -> None:
        super().__init__()

        assert num_radial <= 64
        self.num_spherical = num_spherical
        self.num_radial = num_radial
        self.cutoff = cutoff
        self.envelope = Envelope(envelope_exponent)

        bessel_forms = bessel_basis(num_spherical, num_radial)
        sph_harm_forms = real_sph_harm(num_spherical)
        self.sph_funcs = []
        self.bessel_funcs = []

        x, theta = sp.symbols("x theta")
        modules = {"sin": torch.sin, "cos": torch.cos}
        for i in range(num_spherical):
            if i == 0:
                sph1 = sp.lambdify([theta], sph_harm_forms[i][0], modules)(0)
                self.sph_funcs.append(functools.partial(self._sph_to_tensor, sph1))
            else:
                sph = sp.lambdify([theta], sph_harm_forms[i][0], modules)
                self.sph_funcs.append(sph)
            for j in range(num_radial):
                bessel = sp.lambdify([x], bessel_forms[i][j], modules)
                self.bessel_funcs.append(bessel)

    @staticmethod
    def _sph_to_tensor(sph, x: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(x) + sph

    def forward(
        self,
        dist: torch.Tensor,
        angle: torch.Tensor,
        idx_kj: torch.Tensor,
    ) -> torch.Tensor:
        dist = dist / self.cutoff
        rbf = torch.stack([f(dist) for f in self.bessel_funcs], dim=1)
        rbf = self.envelope(dist).unsqueeze(-1) * rbf

        cbf = torch.stack([f(angle) for f in self.sph_funcs], dim=1)

        n, k = self.num_spherical, self.num_radial
        out = (rbf[idx_kj].view(-1, n, k) * cbf.view(-1, n, 1)).view(-1, n * k)
        return out


class BesselBasisLayer(torch.nn.Module):

    def __init__(
        self,
        num_radial: int,
        cutoff: float = 5.0,
        envelope_exponent: int = 5,
    ) -> None:
        super().__init__()
        self.cutoff = cutoff

        # Ensures that the basis functions smoothly decay to zero at the cutoff distance.
        self.envelope = Envelope(envelope_exponent)

        # A learnable parameter initialized as π * [1, 2, ..., num_radial].
        self.freq = torch.nn.Parameter(torch.empty(num_radial))

    def reset_parameters(self) -> None:
        with torch.no_grad():
            torch.arange(1, self.freq.numel() + 1, out=self.freq).mul_(math.pi)
        self.freq.requires_grad_()

    def forward(self, dist: torch.Tensor) -> torch.Tensor:
        dist = dist.unsqueeze(-1) / self.cutoff
        return self.envelope(dist) * (self.freq * dist).sin()


class Envelope(torch.nn.Module):
    # The function smoothly transitions to zero at x = 1.0 (i.e., dist = cutoff).
    # Controlled decay using a polynomial function.
    def __init__(self, exponent: int) -> None:
        super().__init__()

        # p = exponent + 1 controls the smoothness.
        self.p = exponent + 1

        # a, b, c are precomputed constants to enforce smooth behavior.
        self.a = -(self.p + 1) * (self.p + 2) / 2
        self.b = self.p * (self.p + 2)
        self.c = -self.p * (self.p + 1) / 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        p, a, b, c = self.p, self.a, self.b, self.c
        x_pow_p0 = x.pow(p - 1)
        x_pow_p1 = x_pow_p0 * x
        x_pow_p2 = x_pow_p1 * x
        return (1.0 / x + a * x_pow_p0 + b * x_pow_p1 + c * x_pow_p2) * (x < 1.0).to(x.dtype)


###############################################################################
#################################### UTILS ####################################
###############################################################################


def Jn(r: float, n: int) -> np.floating[Any]:
    return np.sqrt(np.pi / (2 * r)) * scipy.special.jv(n + 0.5, r)


def Jn_zeros(n: int, k: int) -> npt.NDArray[np.float32]:
    zerosj = np.zeros((n, k), dtype="float32")
    zerosj[0] = np.arange(1, k + 1) * np.pi
    points = np.arange(1, k + n) * np.pi
    racines = np.zeros(k + n - 1, dtype="float32")
    for i in range(1, n):
        for j in range(k + n - 1 - i):
            foo = scipy.optimize.brentq(Jn, points[j], points[j + 1], (i,))
            racines[j] = foo
        points = racines
        zerosj[i][:k] = racines[:k]

    return zerosj


def spherical_bessel_formulas(n: int) -> list[sp.Expr]:
    x = sp.symbols("x")

    f = [sp.sin(x) / x]
    a = sp.sin(x) / x
    for i in range(1, n):
        b = sp.diff(a, x) / x
        f += [sp.simplify(b * (-x) ** i)]
        a = sp.simplify(b)
    return f


def bessel_basis(n: int, k: int) -> list[list[sp.Expr]]:
    zeros = Jn_zeros(n, k)
    normalizer = []
    for order in range(n):
        normalizer_tmp = []
        for i in range(k):
            normalizer_tmp += [0.5 * Jn(zeros[order, i], order + 1) ** 2]
        normalizer_tmp = 1 / np.array(normalizer_tmp) ** 0.5
        normalizer += [normalizer_tmp]

    f = spherical_bessel_formulas(n)
    x = sp.symbols("x")
    bess_basis = []
    for order in range(n):
        bess_basis_tmp = []
        for i in range(k):
            bess_basis_tmp += [sp.simplify(normalizer[order][i] * f[order].subs(x, zeros[order, i] * x))]
        bess_basis += [bess_basis_tmp]
    return bess_basis


def sph_harm_prefactor(k: int, m: int) -> float:
    return ((2 * k + 1) * math.factorial(k - abs(m)) / (4 * np.pi * math.factorial(k + abs(m)))) ** 0.5


def associated_legendre_polynomials(k: int, zero_m_only: bool = True) -> list[list[Any]]:
    # Helper function to calculate Y_l^m.
    z = sp.symbols("z")
    P_l_m: list[list[Any]] = [[0] * (j + 1) for j in range(k)]

    P_l_m[0][0] = 1
    if k > 0:
        P_l_m[1][0] = z

        for j in range(2, k):
            # Use the property of Eq (7) in
            # https://mathworld.wolfram.com/AssociatedLegendrePolynomial.html:
            P_l_m[j][0] = sp.simplify(((2 * j - 1) * z * P_l_m[j - 1][0] - (j - 1) * P_l_m[j - 2][0]) / j)
        if not zero_m_only:
            for i in range(1, k):
                P_l_m[i][i] = sp.simplify((1 - 2 * i) * P_l_m[i - 1][i - 1] * (1 - z**2) ** 0.5)
                if i + 1 < k:
                    # Use the property of Eq (11) in
                    # https://mathworld.wolfram.com/AssociatedLegendrePolynomial.html:
                    P_l_m[i + 1][i] = sp.simplify((2 * i + 1) * z * P_l_m[i][i])
                for j in range(i + 2, k):
                    # Use the property of Eq (7) in
                    # https://mathworld.wolfram.com/AssociatedLegendrePolynomial.html:
                    P_l_m[j][i] = sp.simplify(
                        ((2 * j - 1) * z * P_l_m[j - 1][i] - (i + j - 1) * P_l_m[j - 2][i]) / (j - i)
                    )

    return P_l_m


def real_sph_harm(k: int, zero_m_only: bool = True, spherical_coordinates: bool = True) -> list[list[Any]]:
    x = sp.symbols("x")
    y = sp.symbols("y")

    S_m: list[Any] = []
    C_m: list[Any] = []

    if not zero_m_only:
        S_m = [sp.Integer(0)]
        C_m = [sp.Integer(1)]
        for i in range(1, k):
            S_m += [x * S_m[i - 1] + y * C_m[i - 1]]
            C_m += [x * C_m[i - 1] - y * S_m[i - 1]]

    P_l_m = associated_legendre_polynomials(k, zero_m_only)
    if spherical_coordinates:
        theta = sp.symbols("theta")
        z = sp.symbols("z")
        for i in range(len(P_l_m)):
            for j in range(len(P_l_m[i])):
                if not isinstance(P_l_m[i][j], int):
                    P_l_m[i][j] = P_l_m[i][j].subs(z, sp.cos(theta))
        if not zero_m_only:
            phi = sp.symbols("phi")
            for i in range(len(S_m)):
                S_m[i] = S_m[i].subs(x, sp.sin(theta) * sp.cos(phi)).subs(y, sp.sin(theta) * sp.sin(phi))  # type: ignore[operator]
            for i in range(len(C_m)):
                C_m[i] = C_m[i].subs(x, sp.sin(theta) * sp.cos(phi)).subs(y, sp.sin(theta) * sp.sin(phi))  # type: ignore[operator]

    Y_func_l_m: list[list[Any]] = [["0"] * (2 * j + 1) for j in range(k)]
    for i in range(k):
        Y_func_l_m[i][0] = sp.simplify(sph_harm_prefactor(i, 0) * P_l_m[i][0])

    if not zero_m_only:
        for i in range(1, k):
            for j in range(1, i + 1):
                Y_func_l_m[i][j] = sp.simplify(2**0.5 * sph_harm_prefactor(i, j) * C_m[j] * P_l_m[i][j])
        for i in range(1, k):
            for j in range(1, i + 1):
                Y_func_l_m[i][-j] = sp.simplify(2**0.5 * sph_harm_prefactor(i, -j) * S_m[j] * P_l_m[i][j])

    return Y_func_l_m
