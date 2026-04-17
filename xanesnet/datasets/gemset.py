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

import os
from dataclasses import dataclass, fields
from typing import Any

import numba
import numpy as np
import scipy.sparse as sp
import torch
from tqdm import tqdm

from xanesnet.datasources import DataSource
from xanesnet.serialization.config import Config

from .registry import DatasetRegistry
from .torch_dataset import TorchDataset

###############################################################################
################################# DATA CLASSES ################################
###############################################################################


@dataclass
class GemNetData:
    """
    Per-sample data stored on disk.
    """

    num_atoms: torch.Tensor  # scalar
    atomic_numbers: torch.Tensor  # (num_atoms,)
    atom_positions: torch.Tensor  # (num_atoms, 3)
    intensities: torch.Tensor  # (spectrum_size,)
    energies: torch.Tensor  # (spectrum_size,)
    sample_id: str = ""

    def to_state_dict(self) -> dict[str, Any]:
        return {
            "num_atoms": self.num_atoms,
            "atomic_numbers": self.atomic_numbers,
            "atom_positions": self.atom_positions,
            "intensities": self.intensities,
            "energies": self.energies,
            "sample_id": self.sample_id,
        }

    @classmethod
    def from_state_dict(cls, state: dict[str, Any]) -> "GemNetData":
        return cls(
            num_atoms=state["num_atoms"],
            atomic_numbers=state["atomic_numbers"],
            atom_positions=state["atom_positions"],
            intensities=state["intensities"],
            energies=state["energies"],
            sample_id=state.get("sample_id", ""),
        )

    def save(self, path: str) -> str:
        torch.save(self.to_state_dict(), path)
        return path

    @classmethod
    def load(cls, path: str) -> "GemNetData":
        state = torch.load(path, weights_only=True)
        return cls.from_state_dict(state)


@dataclass
class GemNetBatch:
    """
    Batched data with precomputed graph indices for GemNet.
    """

    # Per-atom data (concatenated across batch)
    num_atoms: torch.Tensor  # (nMolecules,)
    atomic_numbers: torch.Tensor  # (nAtoms,)
    atom_positions: torch.Tensor  # (nAtoms, 3)
    batch_seg: torch.Tensor  # (nAtoms,)

    # Per-molecule data (stacked)
    intensities: torch.Tensor  # (nMolecules, spectrum_size)
    energies: torch.Tensor  # (nMolecules, spectrum_size)
    sample_id: list[str]  # (nMolecules,)

    # Edge indices
    id_undir: torch.Tensor  # (nEdges,)
    id_swap: torch.Tensor  # (nEdges,)
    id_c: torch.Tensor  # (nEdges,) source
    id_a: torch.Tensor  # (nEdges,) target

    # Triplet indices
    id3_expand_ba: torch.Tensor  # (nTriplets,)
    id3_reduce_ca: torch.Tensor  # (nTriplets,)
    Kidx3: torch.Tensor  # (nTriplets,)

    # Quadruplet indices (None if triplets_only)
    id4_int_b: torch.Tensor | None = None
    id4_int_a: torch.Tensor | None = None
    id4_reduce_ca: torch.Tensor | None = None
    id4_expand_db: torch.Tensor | None = None
    id4_reduce_cab: torch.Tensor | None = None
    id4_expand_abd: torch.Tensor | None = None
    Kidx4: torch.Tensor | None = None
    id4_reduce_intm_ca: torch.Tensor | None = None
    id4_expand_intm_db: torch.Tensor | None = None
    id4_reduce_intm_ab: torch.Tensor | None = None
    id4_expand_intm_ab: torch.Tensor | None = None

    def to(self, device: str | torch.device) -> "GemNetBatch":
        """
        Send all tensor fields to the given device.
        """
        for f in fields(self):
            val = getattr(self, f.name)
            if isinstance(val, torch.Tensor):
                setattr(self, f.name, val.to(device))
        return self


###############################################################################
#################################### CLASS ####################################
###############################################################################


@DatasetRegistry.register("gemset")
class GemNetDataset(TorchDataset):
    """
    Dataset for GemNet models.

    Pre-processes molecular data from a DataSource and computes graph-level
    indices (edges, triplets, quadruplets) at batch time in the collate function.
    """

    # Index keys used for triplet-only mode
    TRIPLET_INDEX_KEYS = [
        "batch_seg",
        "id_undir",
        "id_swap",
        "id_c",
        "id_a",
        "id3_expand_ba",
        "id3_reduce_ca",
        "Kidx3",
    ]

    # Additional index keys for quadruplet mode
    QUADRUPLET_INDEX_KEYS = [
        "id4_int_b",
        "id4_int_a",
        "id4_reduce_ca",
        "id4_expand_db",
        "id4_reduce_cab",
        "id4_expand_abd",
        "Kidx4",
        "id4_reduce_intm_ca",
        "id4_expand_intm_db",
        "id4_reduce_intm_ab",
        "id4_expand_intm_ab",
    ]

    def __init__(
        self,
        dataset_type: str,
        datasource: DataSource,
        root: str,
        preload: bool,
        force_prepare: bool,
        split_ratios: list[float] | None,
        split_indexfile: str | None,
        # params:
        cutoff: float,
        int_cutoff: float,
        triplets_only: bool = False,
    ) -> None:
        super().__init__(dataset_type, datasource, root, preload, force_prepare, split_ratios, split_indexfile)

        self.cutoff = cutoff
        self.int_cutoff = int_cutoff
        self.triplets_only = triplets_only

    def prepare(self) -> bool:
        already_processed = super().prepare()
        if already_processed:
            return True

        for idx, pmg_obj in tqdm(enumerate(self.datasource), total=len(self.datasource), desc="Processing data"):
            sample_id = pmg_obj.properties["file_name"]

            atomic_numbers = torch.tensor(pmg_obj.atomic_numbers, dtype=torch.int64)
            cart_coords = torch.tensor(pmg_obj.cart_coords, dtype=torch.float32)
            num_atoms = torch.tensor(len(atomic_numbers), dtype=torch.int64)

            # XANES (first atom)
            energies_np, intensities_np = (
                pmg_obj.site_properties["XANES"][0]["energies"],
                pmg_obj.site_properties["XANES"][0]["intensities"],
            )
            energies = torch.tensor(energies_np, dtype=torch.float32)
            spectra = torch.tensor(intensities_np, dtype=torch.float32)

            data = GemNetData(
                num_atoms=num_atoms,
                atomic_numbers=atomic_numbers,
                atom_positions=cart_coords,
                intensities=spectra,
                energies=energies,
                sample_id=sample_id,
            )

            save_path = os.path.join(self.processed_dir, f"{idx}.pth")
            data.save(save_path)

        return True

    def _load_item(self, path: str) -> GemNetData:
        return GemNetData.load(path)

    def collate_fn(self, batch: list[GemNetData]) -> GemNetBatch:
        """
        Collate a list of GemNetData samples into a single GemNetBatch.

        This computes all graph-level indices (edges, triplets, quadruplets)
        required by GemNet at batch time.
        """
        n_molecules = len(batch)
        num_atoms = np.array([int(s.num_atoms.item()) for s in batch], dtype=np.int32)
        cum_atoms = np.concatenate([[0], np.cumsum(num_atoms)])

        # Concatenate per-atom arrays
        atomic_numbers = torch.cat([s.atomic_numbers for s in batch])
        atom_positions = torch.cat([s.atom_positions for s in batch])

        # Stack per-molecule arrays
        spectra = torch.stack([s.intensities for s in batch])
        energies = torch.stack([s.energies for s in batch])
        sample_ids = [s.sample_id for s in batch]

        # Batch segment: which molecule each atom belongs to
        batch_seg = np.repeat(np.arange(n_molecules, dtype=np.int32), num_atoms)

        # ---- Compute adjacency matrices and edge / triplet / quadruplet indices ---- #
        adj_matrices: list[sp.csr_matrix] = []
        adj_matrices_int: list[sp.csr_matrix] = []
        idx_int_t: np.ndarray = np.array([], dtype=np.int32)
        idx_int_s: np.ndarray = np.array([], dtype=np.int32)
        all_positions = atom_positions.numpy()

        for k in range(n_molecules):
            n = num_atoms[k]
            s, e = cum_atoms[k], cum_atoms[k + 1]
            R = all_positions[s:e]

            D_ij = np.linalg.norm(R[:, None, :] - R[None, :, :], axis=-1)

            # Adjacency for embedding edges
            adj_mat = sp.csr_matrix(D_ij <= self.cutoff)
            adj_mat -= sp.eye(n, dtype=bool)
            adj_matrices.append(adj_mat)

            if not self.triplets_only:
                # Adjacency for interaction edges
                adj_mat_int = sp.csr_matrix(D_ij <= self.int_cutoff)
                adj_mat_int -= sp.eye(n, dtype=bool)
                adj_matrices_int.append(adj_mat_int)

        # Combined block-diagonal adjacency
        adj_matrix = _bmat_fast(adj_matrices)
        idx_t, idx_s = adj_matrix.nonzero()  # target and source

        if not self.triplets_only:
            adj_matrix_int = _bmat_fast(adj_matrices_int)
            idx_int_t, idx_int_s = adj_matrix_int.nonzero()

        # Handle empty-edges case
        if len(idx_t) == 0:
            empty = torch.tensor([], dtype=torch.int64)
            return GemNetBatch(
                num_atoms=torch.tensor(num_atoms, dtype=torch.int64),
                atomic_numbers=atomic_numbers,
                atom_positions=atom_positions,
                batch_seg=torch.tensor(batch_seg, dtype=torch.int64),
                intensities=spectra,
                energies=energies,
                sample_id=sample_ids,
                id_undir=empty,
                id_swap=empty,
                id_c=empty,
                id_a=empty,
                id3_expand_ba=empty,
                id3_reduce_ca=empty,
                Kidx3=empty,
            )

        # ---- Reorder edges: undirected pairs ---- #
        edges = np.stack([idx_t, idx_s], axis=0)
        mask = edges[0] < edges[1]
        edges = edges[:, mask]
        edges = np.concatenate([edges, edges[::-1]], axis=-1).astype("int32")
        idx_t, idx_s = edges[0], edges[1]

        N_undir_edges = int(len(idx_s) / 2)
        indices = np.arange(N_undir_edges, dtype="int32")
        id_undir = np.concatenate(2 * [indices], axis=-1).astype("int32")

        id_c = idx_s  # source
        id_a = idx_t  # target

        ind = np.arange(N_undir_edges, dtype="int32")
        id_swap = np.concatenate([ind + N_undir_edges, ind])

        edge_ids = sp.csr_matrix(
            (np.arange(len(idx_s)), (idx_t, idx_s)),
            shape=adj_matrix.shape,
            dtype="int32",
        )

        # ---- Triplets ---- #
        id3_expand_ba, id3_reduce_ca = _get_triplets(idx_s, idx_t, edge_ids)
        id3_reduce_ca = id_swap[id3_reduce_ca]

        if len(id3_reduce_ca) > 0:
            idx_sorted = np.argsort(id3_reduce_ca)
            id3_reduce_ca = id3_reduce_ca[idx_sorted]
            id3_expand_ba = id3_expand_ba[idx_sorted]
            _, K = np.unique(id3_reduce_ca, return_counts=True)
            Kidx3 = _ragged_range(K)
        else:
            Kidx3 = np.array([], dtype="int32")

        # Build quadruplet data (or None)
        quad_tensors: dict[str, torch.Tensor | None] = {k: None for k in self.QUADRUPLET_INDEX_KEYS}

        if not self.triplets_only:
            output = _get_quadruplets(idx_s, idx_t, adj_matrix, edge_ids, idx_int_s, idx_int_t)
            (
                q_id4_reduce_ca,
                q_id4_expand_db,
                q_id4_reduce_cab,
                q_id4_expand_abd,
                q_id4_reduce_intm_ca,
                q_id4_expand_intm_db,
                q_id4_reduce_intm_ab,
                q_id4_expand_intm_ab,
            ) = output

            if len(q_id4_reduce_ca) > 0:
                sorted_idx = np.argsort(q_id4_reduce_ca)
                q_id4_reduce_ca = q_id4_reduce_ca[sorted_idx]
                q_id4_expand_db = q_id4_expand_db[sorted_idx]
                q_id4_reduce_cab = q_id4_reduce_cab[sorted_idx]
                q_id4_expand_abd = q_id4_expand_abd[sorted_idx]
                _, K4 = np.unique(q_id4_reduce_ca, return_counts=True)
                Kidx4 = _ragged_range(K4)
            else:
                Kidx4 = np.array([], dtype="int32")

            quad_tensors = {
                "id4_int_b": torch.tensor(idx_int_s, dtype=torch.int64),
                "id4_int_a": torch.tensor(idx_int_t, dtype=torch.int64),
                "id4_reduce_ca": torch.tensor(q_id4_reduce_ca, dtype=torch.int64),
                "id4_expand_db": torch.tensor(q_id4_expand_db, dtype=torch.int64),
                "id4_reduce_cab": torch.tensor(q_id4_reduce_cab, dtype=torch.int64),
                "id4_expand_abd": torch.tensor(q_id4_expand_abd, dtype=torch.int64),
                "Kidx4": torch.tensor(Kidx4, dtype=torch.int64),
                "id4_reduce_intm_ca": torch.tensor(q_id4_reduce_intm_ca, dtype=torch.int64),
                "id4_expand_intm_db": torch.tensor(q_id4_expand_intm_db, dtype=torch.int64),
                "id4_reduce_intm_ab": torch.tensor(q_id4_reduce_intm_ab, dtype=torch.int64),
                "id4_expand_intm_ab": torch.tensor(q_id4_expand_intm_ab, dtype=torch.int64),
            }

        return GemNetBatch(
            num_atoms=torch.tensor(num_atoms, dtype=torch.int64),
            atomic_numbers=atomic_numbers,
            atom_positions=atom_positions,
            batch_seg=torch.tensor(batch_seg, dtype=torch.int64),
            intensities=spectra,
            energies=energies,
            sample_id=sample_ids,
            id_undir=torch.tensor(id_undir, dtype=torch.int64),
            id_swap=torch.tensor(id_swap, dtype=torch.int64),
            id_c=torch.tensor(id_c, dtype=torch.int64),
            id_a=torch.tensor(id_a, dtype=torch.int64),
            id3_expand_ba=torch.tensor(id3_expand_ba, dtype=torch.int64),
            id3_reduce_ca=torch.tensor(id3_reduce_ca, dtype=torch.int64),
            Kidx3=torch.tensor(Kidx3, dtype=torch.int64),
            **quad_tensors,
        )

    @property
    def signature(self) -> Config:
        """
        Return dataset signature as a dictionary.
        """
        signature = super().signature
        signature.update_with_dict(
            {
                "cutoff": self.cutoff,
                "int_cutoff": self.int_cutoff,
                "triplets_only": self.triplets_only,
            }
        )
        return signature


###############################################################################
############################# HELPER FUNCTIONS ################################
###############################################################################


def _bmat_fast(mats: list[sp.csr_matrix]) -> sp.csr_matrix:
    """
    Combine multiple adjacency matrices into a single sparse block-diagonal matrix.
    """
    assert len(mats) > 0
    new_data = np.concatenate([mat.data for mat in mats])  # type: ignore[union-attr]

    ind_offset = np.zeros(1 + len(mats), dtype="int32")
    ind_offset[1:] = np.cumsum([mat.shape[0] for mat in mats])  # type: ignore[union-attr]
    new_indices = np.concatenate([mats[i].indices + ind_offset[i] for i in range(len(mats))])

    indptr_offset = np.zeros(1 + len(mats))
    indptr_offset[1:] = np.cumsum([mat.nnz for mat in mats])
    new_indptr = np.concatenate([mats[i].indptr[i >= 1 :] + indptr_offset[i] for i in range(len(mats))])

    shape = (ind_offset[-1], ind_offset[-1])

    if len(new_data) == 0:
        return sp.csr_matrix(shape)

    return sp.csr_matrix((new_data, new_indices, new_indptr), shape=shape)


def _get_triplets(idx_s: np.ndarray, idx_t: np.ndarray, edge_ids: sp.csr_matrix) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute triplet indices c -> a <- b.
    """
    id3_expand_ba = edge_ids[idx_s].data.astype("int32").flatten()
    id3_reduce_ca = edge_ids[idx_s].tocoo().row.astype("int32").flatten()

    id3_i = idx_t[id3_reduce_ca]
    id3_k = idx_s[id3_expand_ba]
    mask = id3_i != id3_k
    id3_expand_ba = id3_expand_ba[mask]
    id3_reduce_ca = id3_reduce_ca[mask]

    return id3_expand_ba, id3_reduce_ca


def _get_quadruplets(
    idx_s: np.ndarray,
    idx_t: np.ndarray,
    adj_matrix: sp.csr_matrix,
    edge_ids: sp.csr_matrix,
    idx_int_s: np.ndarray,
    idx_int_t: np.ndarray,
) -> tuple[np.ndarray, ...]:
    """
    Compute quadruplet indices c -> a - b <- d
    where D_ab <= int_cutoff; D_ca & D_db <= cutoff.
    """
    nNeighbors_t = adj_matrix[idx_int_t].sum(axis=1).A1.astype("int32")
    nNeighbors_s = adj_matrix[idx_int_s].sum(axis=1).A1.astype("int32")
    id4_reduce_intm_ca = edge_ids[idx_int_t].data.astype("int32").flatten()
    id4_expand_intm_db = edge_ids[idx_int_s].data.astype("int32").flatten()

    id4_reduce_cab = _repeat_blocks(nNeighbors_t, nNeighbors_s)
    id4_reduce_ca = id4_reduce_intm_ca[id4_reduce_cab]

    N = np.repeat(nNeighbors_t, nNeighbors_s)
    id4_expand_abd = np.repeat(np.arange(len(id4_expand_intm_db)), N)
    id4_expand_db = id4_expand_intm_db[id4_expand_abd]

    id4_reduce_intm_ab = np.repeat(np.arange(len(idx_int_t)), nNeighbors_t)
    id4_expand_intm_ab = np.repeat(np.arange(len(idx_int_t)), nNeighbors_s)

    # Mask out quadruplets where nodes appear more than once
    idx_c = idx_s[id4_reduce_ca]
    idx_a = idx_t[id4_reduce_ca]
    idx_b = idx_t[id4_expand_db]
    idx_d = idx_s[id4_expand_db]

    mask = (idx_c != idx_b) & (idx_a != idx_d) & (idx_c != idx_d)

    id4_reduce_ca = id4_reduce_ca[mask]
    id4_expand_db = id4_expand_db[mask]
    id4_reduce_cab = id4_reduce_cab[mask]
    id4_expand_abd = id4_expand_abd[mask]

    return (
        id4_reduce_ca,
        id4_expand_db,
        id4_reduce_cab,
        id4_expand_abd,
        id4_reduce_intm_ca,
        id4_expand_intm_db,
        id4_reduce_intm_ab,
        id4_expand_intm_ab,
    )


@numba.njit(nogil=True)
def _repeat_blocks(sizes: np.ndarray, repeats: np.ndarray) -> np.ndarray:
    """
    Repeat blocks of consecutive indices.

    Example: sizes=[1,3,2], repeats=[3,2,3]
      -> [0 0 0  1 2 3 1 2 3  4 5 4 5 4 5]
    """
    a = np.arange(np.sum(sizes))
    indices = np.empty((sizes * repeats).sum(), dtype=np.int32)
    start = 0
    oi = 0
    for i, size in enumerate(sizes):
        end = start + size
        for _ in range(repeats[i]):
            oe = oi + size
            indices[oi:oe] = a[start:end]
            oi = oe
        start = end
    return indices


@numba.njit(nogil=True)
def _ragged_range(sizes: np.ndarray) -> np.ndarray:
    """
    Ragged range: sizes=[1,3,2] -> [0, 0 1 2, 0 1]
    """
    a = np.arange(sizes.max())
    indices = np.empty(sizes.sum(), dtype=np.int32)
    start = 0
    for size in sizes:
        end = start + size
        indices[start:end] = a[:size]
        start = end
    return indices
