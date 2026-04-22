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

import numpy as np
import torch
from pymatgen.core import Molecule, Structure


def _absorber_neighbors(
    pmg_obj: Structure | Molecule,
    absorber_idx: int,
    cutoff: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return (neighbor_indices, neighbor_coords) of the absorber within cutoff.

    For periodic Structures, uses pymatgen's PBC-aware neighbor search so that
    neighbor_coords refer to the correct periodic image positions. For
    molecules, uses Cartesian distances directly.
    """
    abs_coord = np.array(pmg_obj.cart_coords[absorber_idx], dtype=np.float64)

    if isinstance(pmg_obj, Structure):
        neighbors = pmg_obj.get_neighbors(pmg_obj[absorber_idx], r=cutoff)
        if len(neighbors) == 0:
            return (
                np.zeros(0, dtype=np.int64),
                np.zeros((0, 3), dtype=np.float64),
            )
        idx = np.array([n.index for n in neighbors], dtype=np.int64)
        coords = np.array([n.coords for n in neighbors], dtype=np.float64)
        return idx, coords

    # Molecule: filter by Euclidean distance, exclude absorber itself.
    all_coords = np.array(pmg_obj.cart_coords, dtype=np.float64)
    dists = np.linalg.norm(all_coords - abs_coord, axis=-1)
    mask = (dists <= cutoff) & (np.arange(len(pmg_obj)) != absorber_idx)
    idx = np.where(mask)[0].astype(np.int64)
    coords = all_coords[idx]
    return idx, coords


def build_absorber_paths(
    pmg_obj: Structure | Molecule,
    absorber_idx: int,
    cutoff: float,
    max_paths: int,
) -> dict[str, torch.Tensor]:
    """
    Enumerate absorber-centred 3-body paths (absorber, j, k) with j, k within
    cutoff of the absorber. For periodic structures, j and k may be periodic
    images whose scalar geometry is computed from their image Cartesian
    coordinates (minimum-image distances are implicit in pymatgen's neighbor
    search).

    Paths are ordered by ascending (r0j + r0k + 0.5 * rjk) and truncated to
    ``max_paths`` per structure.

    Returns a dict of tensors:
        path_j        [P] int64   absorber-local atom index of j
        path_k        [P] int64   absorber-local atom index of k
        path_r0j      [P] float32 absorber-j distance
        path_r0k      [P] float32 absorber-k distance
        path_rjk      [P] float32 j-k distance
        path_cosangle [P] float32 cos(angle at the absorber)
    """
    neigh_idx, neigh_coords = _absorber_neighbors(pmg_obj, absorber_idx, cutoff)
    abs_coord = np.array(pmg_obj.cart_coords[absorber_idx], dtype=np.float64)

    n = neigh_idx.shape[0]
    if n < 2:
        return {
            "path_j": torch.zeros(0, dtype=torch.int64),
            "path_k": torch.zeros(0, dtype=torch.int64),
            "path_r0j": torch.zeros(0, dtype=torch.float32),
            "path_r0k": torch.zeros(0, dtype=torch.float32),
            "path_rjk": torch.zeros(0, dtype=torch.float32),
            "path_cosangle": torch.zeros(0, dtype=torch.float32),
        }

    # Enumerate ordered index pairs (j < k over the neighbor list ordering).
    ii, jj = np.triu_indices(n, k=1)

    cj = neigh_coords[ii]  # [P, 3]
    ck = neigh_coords[jj]  # [P, 3]
    vj = cj - abs_coord
    vk = ck - abs_coord
    vjk = ck - cj

    r0j = np.linalg.norm(vj, axis=-1)
    r0k = np.linalg.norm(vk, axis=-1)
    rjk = np.linalg.norm(vjk, axis=-1)

    uj = vj / np.clip(r0j, 1e-8, None)[:, None]
    uk = vk / np.clip(r0k, 1e-8, None)[:, None]
    cosang = np.clip((uj * uk).sum(axis=-1), -1.0, 1.0)

    # Truncate to max_paths by importance.
    score = r0j + r0k + 0.5 * rjk
    order = np.argsort(score)
    if order.shape[0] > max_paths:
        order = order[:max_paths]

    ii_sel = ii[order]
    jj_sel = jj[order]

    return {
        "path_j": torch.tensor(neigh_idx[ii_sel], dtype=torch.int64),
        "path_k": torch.tensor(neigh_idx[jj_sel], dtype=torch.int64),
        "path_r0j": torch.tensor(r0j[order], dtype=torch.float32),
        "path_r0k": torch.tensor(r0k[order], dtype=torch.float32),
        "path_rjk": torch.tensor(rjk[order], dtype=torch.float32),
        "path_cosangle": torch.tensor(cosang[order], dtype=torch.float32),
    }
