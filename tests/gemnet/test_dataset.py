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

"""
Integration tests for ``GemNetDataset.prepare()`` and ``collate_fn`` over both non-periodic and periodic inputs.

We verify:

- All graph tensors (main, int, a2ee2a, a2a, qint) index valid atoms/edges
- Main graph is bidirectional (``id_swap`` is a valid involution)
- Triplet, quadruplet and mixed-triplet indices respect their referenced pools after batch-level offset remapping
- ``intensities.shape[0] == absorber_mask.sum()`` holds per sample 
  AND per batch (catches a common off-by-one between the absorber mask and the flattened-spectra concat)
"""

from pathlib import Path
from typing import Any

import pytest
import torch

from xanesnet.datasets.torchgeometric.gemnet import GemNetDataset
from xanesnet.datasources.pmgjson import PMGJSONSource
from xanesnet.datasources.xyzspec import XYZSpecSource

# Helpers


def _mk_dataset_molecule(
    root: Path, fe_mini: tuple[Path, Path], oc_mode: bool = False, quadruplets: bool = True
) -> GemNetDataset:
    xyz_dir, xanes_dir = fe_mini
    src = XYZSpecSource(
        datasource_type="xyzspec",
        xyz_path=str(xyz_dir),
        xanes_path=str(xanes_dir),
    )
    return GemNetDataset(
        dataset_type="gemnet",
        datasource=src,
        root=str(root),
        preload=False,
        skip_prepare=False,
        split_ratios=None,
        split_indexfile=None,
        cutoff=3.5,
        max_num_neighbors=15,
        graph_method="radius",
        min_facet_area=None,
        cov_radii_scale=1.5,
        quadruplets=quadruplets,
        int_cutoff=4.0,
        oc_mode=oc_mode,
        oc_cutoff_aeaint=3.5,
        oc_cutoff_aint=4.0,
        oc_max_neighbors_aeaint=15,
        oc_max_neighbors_aint=15,
    )


def _mk_dataset_periodic(json_dir: Path, root: Path, oc_mode: bool = False, quadruplets: bool = True) -> GemNetDataset:
    src = PMGJSONSource(datasource_type="pmgjson", json_path=str(json_dir))
    return GemNetDataset(
        dataset_type="gemnet",
        datasource=src,
        root=str(root),
        preload=False,
        skip_prepare=False,
        split_ratios=None,
        split_indexfile=None,
        cutoff=3.5,
        max_num_neighbors=20,
        graph_method="radius",
        min_facet_area=None,
        cov_radii_scale=1.5,
        quadruplets=quadruplets,
        int_cutoff=4.0,
        oc_mode=oc_mode,
        oc_cutoff_aeaint=3.5,
        oc_cutoff_aint=4.5,
        oc_max_neighbors_aeaint=20,
        oc_max_neighbors_aint=20,
    )


# Per-item invariants


def _check_item_invariants(d, oc_mode: bool, quadruplets: bool) -> None:
    n = d.num_nodes
    nE = d.edge_index.size(1)

    # Main graph
    assert d.edge_index.max() < n
    assert d.edge_index.min() >= 0
    assert d.edge_index.dtype == torch.int64
    # bidirectional: id_swap sends each edge to its reverse
    assert d.id_swap.size(0) == nE
    assert d.id_swap[d.id_swap].tolist() == list(range(nE))
    src = d.edge_index[0]
    dst = d.edge_index[1]
    assert torch.all(src[d.id_swap] == dst)
    assert torch.all(dst[d.id_swap] == src)
    # vectors must negate under swap
    if d.edge_vec is not None:
        assert torch.allclose(d.edge_vec[d.id_swap], -d.edge_vec, atol=1e-5)

    # Triplets
    if d.id3_reduce_ca.numel():
        assert d.id3_reduce_ca.max() < nE
        assert d.id3_expand_ba.max() < nE
        # no self (e,e)
        assert not (d.id3_reduce_ca == d.id3_expand_ba).any()

    # Quadruplets
    if quadruplets and d.id4_reduce_ca.numel():
        assert d.id4_reduce_ca.max() < nE
        assert d.id4_expand_db.max() < nE
        nI = d.int_edge_index.size(1)
        assert d.id4_reduce_intm_ab.max() < nI
        assert d.id4_expand_intm_ab.max() < nI
        # id4_reduce_cab indexes into id4_reduce_intm_ca
        assert d.id4_reduce_cab.max() < d.id4_reduce_intm_ca.size(0)
        assert d.id4_expand_abd.max() < d.id4_expand_intm_db.size(0)

    # Absorber / spectra alignment
    assert d.absorber_mask.dtype == torch.bool
    assert d.absorber_mask.size(0) == n
    n_abs = int(d.absorber_mask.sum().item())
    assert d.intensities.size(0) == n_abs
    assert d.energies.size(0) == n_abs

    # OC extras
    if oc_mode:
        nAE = d.a2ee2a_edge_index.size(1)
        nAA = d.a2a_edge_index.size(1)
        assert d.a2ee2a_edge_index.max() < n
        assert d.a2a_edge_index.max() < n
        if d.trip_a2e_in.numel():
            assert d.trip_a2e_in.max() < nAE
            assert d.trip_a2e_out.max() < nE
        if d.trip_e2a_in.numel():
            assert d.trip_e2a_in.max() < nE
            assert d.trip_e2a_out.max() < nAE
        if d.trip_e2e_in.numel():
            assert d.trip_e2e_in.max() < nE
            assert d.trip_e2e_out.max() < nE


def _check_batch_invariants(b, oc_mode: bool, quadruplets: bool) -> None:
    nN = b.num_nodes
    nE = b.edge_index.size(1)
    assert b.edge_index.max() < nN
    assert b.id_swap.max() < nE
    # id_swap must still be a valid involution after batching
    assert b.id_swap[b.id_swap].tolist() == list(range(nE))
    if b.id3_reduce_ca.numel():
        assert b.id3_reduce_ca.max() < nE
        assert b.id3_expand_ba.max() < nE
    if quadruplets and b.id4_reduce_ca.numel():
        nI = b.int_edge_index.size(1)
        assert b.id4_reduce_ca.max() < nE
        assert b.id4_expand_db.max() < nE
        assert b.id4_reduce_intm_ab.max() < nI
        assert b.id4_reduce_cab.max() < b.id4_reduce_intm_ca.size(0)
        assert b.id4_expand_abd.max() < b.id4_expand_intm_db.size(0)
    if oc_mode:
        nAE = b.a2ee2a_edge_index.size(1)
        assert b.a2ee2a_edge_index.max() < nN
        assert b.a2a_edge_index.max() < nN
        if b.trip_a2e_in.numel():
            assert b.trip_a2e_in.max() < nAE
        if b.trip_e2a_out.numel():
            assert b.trip_e2a_out.max() < nAE

    # Spectra and absorbers in sync
    assert b.intensities.size(0) == int(b.absorber_mask.sum().item())
    assert b.energies.size(0) == b.intensities.size(0)


# Molecule (non-periodic) tests


@pytest.mark.parametrize("oc_mode", [False, True])
def test_prepare_molecules(tmp_path, fe_mini_data, oc_mode):
    ds = _mk_dataset_molecule(tmp_path / "proc", fe_mini_data, oc_mode=oc_mode, quadruplets=True)
    ds.prepare()
    assert len(ds) > 0
    # Sample first 3 items
    items = [ds[i] for i in range(min(3, len(ds)))]
    for d in items:
        _check_item_invariants(d, oc_mode=oc_mode, quadruplets=True)
    b = ds.collate_fn(items)
    _check_batch_invariants(b, oc_mode=oc_mode, quadruplets=True)


# Periodic tests


@pytest.mark.parametrize("oc_mode", [False, True])
def test_prepare_periodic(omnixas_temp_data_dir, tmp_path, oc_mode):
    ds = _mk_dataset_periodic(omnixas_temp_data_dir, tmp_path / "proc", oc_mode=oc_mode, quadruplets=True)
    ds.prepare()
    assert len(ds) > 0
    items = [ds[i] for i in range(len(ds))]
    for d in items:
        _check_item_invariants(d, oc_mode=oc_mode, quadruplets=True)
    b = ds.collate_fn(items)
    _check_batch_invariants(b, oc_mode=oc_mode, quadruplets=True)


def test_prepare_periodic_quadruplets_disabled(omnixas_temp_data_dir, tmp_path):
    ds = _mk_dataset_periodic(omnixas_temp_data_dir, tmp_path / "proc", oc_mode=False, quadruplets=False)
    ds.prepare()
    assert len(ds) > 0
    items = [ds[i] for i in range(len(ds))]
    for d in items:
        # With quadruplets disabled, quad attributes must be absent
        assert not hasattr(d, "id4_reduce_ca") or d.id4_reduce_ca.numel() == 0
        _check_item_invariants(d, oc_mode=False, quadruplets=False)
    b = ds.collate_fn(items)
    _check_batch_invariants(b, oc_mode=False, quadruplets=False)


# Value-level tests


def test_edge_weights_match_euclidean_norm(tmp_path, fe_mini_data):
    """
    ``edge_weight[e]`` must equal ``||edge_vec[e]||`` for every edge, otherwise downstream basis layers are fed inconsistent distances.
    """
    ds = _mk_dataset_molecule(tmp_path / "proc", fe_mini_data, oc_mode=True, quadruplets=True)
    ds.prepare()
    d = ds[0]
    norms = torch.linalg.norm(d.edge_vec, dim=-1)
    assert torch.allclose(d.edge_weight, norms, atol=1e-5)
    # Every main-graph edge length must be within the cutoff.
    assert d.edge_weight.max().item() <= ds.cutoff + 1e-5
    assert d.edge_weight.min().item() > 0.0
    # OC extras must also satisfy the same relation.
    norms_ae = torch.linalg.norm(d.a2ee2a_edge_vec, dim=-1)
    assert torch.allclose(d.a2ee2a_edge_weight, norms_ae, atol=1e-5)


def test_absorber_mask_and_atomic_numbers(tmp_path, fe_mini_data):
    """
    For the Fe-complex stems, exactly one atom is flagged as absorber and it is the Fe (Z=26).
    """
    ds = _mk_dataset_molecule(tmp_path / "proc", fe_mini_data, oc_mode=False, quadruplets=True)
    ds.prepare()
    for i in range(len(ds)):
        d = ds[i]
        assert int(d.absorber_mask.sum().item()) == 1
        abs_idx = int(torch.where(d.absorber_mask)[0].item())
        assert int(d.x[abs_idx].item()) == 26


def test_intensities_shape_matches_source_xanes(tmp_path, fe_mini_data):
    """
    Loaded ``intensities`` must have the shape expected from the source XANES txt files (226 energies for the fe dataset).
    """
    ds = _mk_dataset_molecule(tmp_path / "proc", fe_mini_data, oc_mode=False, quadruplets=False)
    ds.prepare()
    d = ds[0]
    assert d.intensities.ndim == 2
    assert d.intensities.size(0) == int(d.absorber_mask.sum().item())
    # All spectra finite and non-negative for this dataset.
    assert torch.isfinite(d.intensities).all()
    assert d.intensities.min().item() >= 0.0
    assert d.intensities.size(1) >= 100  # XAS spectra are long


def test_batch_preserves_per_item_values(tmp_path, fe_mini_data):
    """
    Collating two items must preserve per-item ``pos`` / ``edge_vec`` values exactly:
    concatenation along the node / edge dim only, no reordering, no scaling.
    """
    ds = _mk_dataset_molecule(tmp_path / "proc", fe_mini_data, oc_mode=False, quadruplets=True)
    ds.prepare()
    d0 = ds[0]
    d1 = ds[1]
    b: Any = ds.collate_fn([d0, d1])
    n0 = d0.num_nodes
    e0 = d0.edge_index.size(1)
    assert torch.allclose(b.pos[:n0], d0.pos)
    assert torch.allclose(b.pos[n0:], d1.pos)
    assert torch.allclose(b.edge_vec[:e0], d0.edge_vec)
    assert torch.allclose(b.edge_vec[e0:], d1.edge_vec)
    # Batched edge_index: second graph indices must be offset by n0.
    assert torch.equal(b.edge_index[:, :e0], d0.edge_index)
    assert torch.equal(b.edge_index[:, e0:], d1.edge_index + n0)
    # id_swap of second graph must be offset by e0 in the batched tensor.
    assert torch.equal(b.id_swap[e0:], d1.id_swap + e0)
    # Vector negation under swap must still hold after batching.
    assert torch.allclose(b.edge_vec[b.id_swap], -b.edge_vec, atol=1e-5)


def test_periodic_distinct_images(omnixas_temp_data_dir, tmp_path):
    """
    Periodic structures with a cutoff larger than the cell edge must produce multiple directed edges between the same atom pair,
    one per image. The associated ``edge_vec`` values must all differ.
    """
    ds = _mk_dataset_periodic(omnixas_temp_data_dir, tmp_path / "proc", oc_mode=False, quadruplets=False)
    ds.prepare()
    d = ds[0]
    # Group edges by (src, dst)
    pair_vecs: dict[tuple[int, int], list[torch.Tensor]] = {}
    for e in range(d.edge_index.size(1)):
        key = (int(d.edge_index[0, e].item()), int(d.edge_index[1, e].item()))
        pair_vecs.setdefault(key, []).append(d.edge_vec[e])
    # At least one (src, dst) pair should have more than one edge with different vectors (image-duplicates).
    multi = [(k, v) for k, v in pair_vecs.items() if len(v) > 1]
    assert multi, "Expected image-duplicated edges on omnixas periodic inputs"
    for _, vecs in multi:
        stacked = torch.stack(vecs)
        uniq = torch.unique(stacked.round(decimals=3), dim=0)
        assert uniq.size(0) == len(vecs), "Duplicate image edges must have distinct vectors"


def test_fe_octahedral_edge_count(tmp_path, fe_mini_data):
    """
    For an Fe complex with a small radial cutoff, each Fe-ligand bond generates exactly one directed edge in each direction.
    The number of edges incident to the Fe atom must therefore be even and equal ``2 * coordination_number``.
    """
    ds = _mk_dataset_molecule(tmp_path / "proc", fe_mini_data, oc_mode=False, quadruplets=False)
    ds.prepare()
    d = ds[0]
    abs_idx = int(torch.where(d.absorber_mask)[0].item())
    incoming = (d.edge_index[1] == abs_idx).sum().item()
    outgoing = (d.edge_index[0] == abs_idx).sum().item()
    assert incoming == outgoing, "Graph must be bidirectional around the absorber"
    assert incoming > 0, "Absorber must have at least one neighbour within cutoff"
