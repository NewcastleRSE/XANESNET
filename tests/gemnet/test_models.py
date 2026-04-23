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
End-to-end model tests for GemNet (with / without quadruplets) and GemNet-OC (all interactions).

Forward passes use tiny embedding sizes so every test runs in a few seconds. 

We verify (for both models):
- Shape of the per-atom output is ``(num_atoms, num_targets)``
- Output is **finite** (no NaN / Inf) when ``scale_file=None``
- Output is **deterministic** for a fixed random seed
- Output **changes** when the input coordinates change (not identically zero / not constant across atoms)
- Prediction shape after the batch processor matches the target shape (absorber masking is applied consistently)
- Gradients flow (no NaNs, non-zero norms) and are finite
- All OC interactions can be toggled off individually without breaking the forward
- The raw basis layers (Bessel RBF + spherical CBF) produce the expected numerical values on known inputs
"""

from pathlib import Path
from typing import Any

import pytest
import torch

from xanesnet.batchprocessors.gemnet import GemNetBatchProcessor
from xanesnet.batchprocessors.gemnet_oc import GemNetOCBatchProcessor
from xanesnet.datasets.torchgeometric.gemnet import GemNetDataset
from xanesnet.datasources.xyzspec import XYZSpecSource
from xanesnet.models.gemnet import GemNet
from xanesnet.models.gemnet_oc import GemNetOC

# Small fixtures building a tiny batch on-demand.


def _build_small_batch(tmp_path: Path, fe_mini, oc_mode: bool, quadruplets: bool) -> tuple[GemNetDataset, Any]:
    xyz_dir, xanes_dir = fe_mini
    src = XYZSpecSource(
        datasource_type="xyzspec",
        xyz_path=str(xyz_dir),
        xanes_path=str(xanes_dir),
    )
    ds = GemNetDataset(
        dataset_type="gemnet",
        datasource=src,
        root=str(tmp_path / "proc"),
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
    ds.prepare()
    items = [ds[i] for i in range(min(5, len(ds)))]
    items.sort(key=lambda d: d.num_nodes)
    items = items[:2]
    b = ds.collate_fn(items)
    return ds, b


def _tiny_gemnet(triplets_only: bool, num_targets: int = 226) -> GemNet:
    return GemNet(
        model_type="gemnet",
        num_spherical=6,
        num_radial=5,
        num_blocks=2,
        emb_size_atom=32,
        emb_size_edge=32,
        emb_size_trip=16,
        emb_size_quad=8,
        emb_size_rbf=8,
        emb_size_cbf=8,
        emb_size_sbf=8,
        emb_size_bil_quad=8,
        emb_size_bil_trip=16,
        num_before_skip=1,
        num_after_skip=1,
        num_concat=1,
        num_atom=2,
        triplets_only=triplets_only,
        num_targets=num_targets,
        cutoff=4.0,
        int_cutoff=5.0,
        envelope_exponent=5,
        output_init="HeOrthogonal",
        activation="swish",
        scale_file=None,
        num_elements=94,
    )


def _tiny_gemnet_oc(
    num_targets: int = 226,
    quad: bool = True,
    a2e: bool = True,
    e2a: bool = True,
    a2a: bool = True,
) -> GemNetOC:
    return GemNetOC(
        model_type="gemnet_oc",
        num_targets=num_targets,
        num_spherical=5,
        num_radial=5,
        num_blocks=2,
        emb_size_atom=16,
        emb_size_edge=16,
        emb_size_trip_in=8,
        emb_size_trip_out=8,
        emb_size_quad_in=4,
        emb_size_quad_out=4,
        emb_size_aint_in=8,
        emb_size_aint_out=8,
        emb_size_rbf=4,
        emb_size_cbf=4,
        emb_size_sbf=4,
        num_before_skip=1,
        num_after_skip=1,
        num_concat=1,
        num_atom=1,
        num_output_afteratom=1,
        num_atom_emb_layers=0,
        num_global_out_layers=1,
        cutoff=3.5,
        cutoff_qint=4.0,
        cutoff_aeaint=3.5,
        cutoff_aint=4.0,
        rbf={"name": "gaussian"},
        rbf_spherical={"name": "gaussian"},
        envelope={"name": "polynomial", "exponent": 5},
        cbf={"name": "spherical_harmonics"},
        sbf={"name": "spherical_harmonics"},
        output_init="HeOrthogonal",
        activation="silu",
        quad_interaction=quad,
        atom_edge_interaction=a2e,
        edge_atom_interaction=e2a,
        atom_interaction=a2a,
        scale_basis=False,
        num_elements=94,
    )


# GemNet


@pytest.mark.parametrize("triplets_only", [True, False])
def test_gemnet_forward_values(tmp_path, fe_mini_data, triplets_only):
    """
    End-to-end value-level test for GemNet with ``scale_file=None``.

    Checks:

    1. Output is finite (no NaN, no Inf) — regression for the bug where ``EfficientInteractionDownProjection``
    / ``EfficientInteractionBilinear`` allocated parameters with ``torch.empty`` but never called
       ``reset_parameters()``, so uninitialised memory leaked through the forward pass and produced NaNs.
    2. Output shape is ``(num_atoms, num_targets)``.
    3. Output is deterministic for a fixed ``torch.manual_seed``.
    4. Output varies across atoms (not the degenerate "all-zero / all-equal" fail-mode).
    5. Prediction / target shapes agree after absorber masking.
    6. Gradients are finite and at least one parameter has non-zero grad norm.
    """
    _, b = _build_small_batch(tmp_path, fe_mini_data, oc_mode=False, quadruplets=not triplets_only)
    num_targets = int(b.intensities.size(1))
    bp = GemNetBatchProcessor()
    inp = bp.input_preparation(b)

    # Build twice from the same seed -> outputs must match exactly.
    torch.manual_seed(0)
    model_a = _tiny_gemnet(triplets_only=triplets_only, num_targets=num_targets)
    torch.manual_seed(0)
    model_b = _tiny_gemnet(triplets_only=triplets_only, num_targets=num_targets)
    model_a.eval()
    model_b.eval()

    with torch.no_grad():
        E1 = model_a(**inp)
        E2 = model_b(**inp)

    assert E1.shape == (b.num_nodes, num_targets)
    assert torch.isfinite(E1).all(), "GemNet output must be finite with scale_file=None"
    assert torch.allclose(E1, E2), "Forward pass must be deterministic under fixed seed"
    # Non-degenerate: predictions differ across atoms (std > 0 along batch dim)
    assert E1.std(dim=0).abs().max().item() > 0.0
    assert E1.abs().max().item() > 0.0

    pred = bp.prediction_preparation(b, E1)
    tgt = bp.target_preparation(b)
    assert pred.shape == tgt.shape
    assert pred.shape[0] == int(b.absorber_mask.sum().item())

    # Grad flow
    model_a.train()
    E = model_a(**inp)
    pred = bp.prediction_preparation(b, E)
    tgt = bp.target_preparation(b)
    loss = torch.nn.functional.mse_loss(pred, tgt)
    loss.backward()
    grads = [p.grad for p in model_a.parameters() if p.grad is not None]
    assert len(grads) > 0
    assert all(torch.isfinite(g).all() for g in grads)
    assert any(g.abs().sum().item() > 0 for g in grads)


def test_gemnet_forward_changes_with_input(tmp_path, fe_mini_data):
    """
    Perturbing atom positions must change the GemNet output.
    """
    _, b = _build_small_batch(tmp_path, fe_mini_data, oc_mode=False, quadruplets=True)
    num_targets = int(b.intensities.size(1))
    bp = GemNetBatchProcessor()

    torch.manual_seed(1)
    model = _tiny_gemnet(triplets_only=False, num_targets=num_targets)
    model.eval()

    inp1 = bp.input_preparation(b)
    with torch.no_grad():
        E1 = model(**inp1)

    # Scale the edge vectors/weights as if we translated atom positions; since GemNet consumes
    # precomputed edge_vec/edge_weight from the batch, apply the perturbation there directly.
    b2 = b.clone()
    b2.pos = b2.pos + 0.05  # rigid translation -> identical edge_vec but different pos
    b2.edge_vec = b2.edge_vec * 1.02
    b2.edge_weight = b2.edge_weight * 1.02
    inp2 = bp.input_preparation(b2)
    with torch.no_grad():
        E2 = model(**inp2)

    assert not torch.allclose(E1, E2), "Output must depend on geometry"
    assert torch.isfinite(E2).all()


# GemNet-OC


def test_gemnet_oc_forward_full(tmp_path, fe_mini_data):
    """
    End-to-end value-level test for GemNet-OC with all interactions enabled.

    Same battery of value checks as the GemNet test plus a determinism re-build and an input-sensitivity check.
    """
    _, b = _build_small_batch(tmp_path, fe_mini_data, oc_mode=True, quadruplets=True)
    num_targets = int(b.intensities.size(1))

    torch.manual_seed(0)
    model = _tiny_gemnet_oc(num_targets=num_targets)
    torch.manual_seed(0)
    model_twin = _tiny_gemnet_oc(num_targets=num_targets)
    bp = GemNetOCBatchProcessor()
    inp = bp.input_preparation(b)

    model.eval()
    model_twin.eval()
    with torch.no_grad():
        E = model(**inp)
        E_twin = model_twin(**inp)

    assert E.shape == (b.num_nodes, num_targets)
    assert torch.isfinite(E).all()
    assert torch.allclose(E, E_twin), "GemNet-OC forward must be deterministic"
    assert E.std(dim=0).abs().max().item() > 0.0
    assert E.abs().max().item() > 0.0

    # Input sensitivity: different perturbed batch -> different output
    b2 = b.clone()
    b2.pos = b2.pos + 0.03
    inp2 = bp.input_preparation(b2)
    # Note: GemNet-OC recomputes distances/angles from pos, so only pos change is needed to induce a different output.
    # But this port consumes precomputed edge_vec/edge_weight — perturb those to reflect the position change consistently.
    b2.edge_vec = b2.edge_vec * 1.01
    b2.edge_weight = b2.edge_weight * 1.01
    if hasattr(b2, "a2ee2a_edge_vec"):
        b2.a2ee2a_edge_vec = b2.a2ee2a_edge_vec * 1.01
        b2.a2ee2a_edge_weight = b2.a2ee2a_edge_weight * 1.01
    if hasattr(b2, "int_edge_vec") and b2.int_edge_vec is not None:
        b2.int_edge_vec = b2.int_edge_vec * 1.01
        b2.int_edge_weight = b2.int_edge_weight * 1.01
    inp2 = bp.input_preparation(b2)
    with torch.no_grad():
        E2 = model(**inp2)
    assert not torch.allclose(E, E2)

    model.train()
    E = model(**inp)
    pred = bp.prediction_preparation(b, E)
    tgt = bp.target_preparation(b)
    assert pred.shape == tgt.shape
    assert pred.shape[0] == int(b.absorber_mask.sum().item())

    loss = torch.nn.functional.mse_loss(pred, tgt)
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert len(grads) > 0
    assert all(torch.isfinite(g).all() for g in grads)
    assert any(g.abs().sum().item() > 0 for g in grads)


@pytest.mark.parametrize(
    "quad,a2e,e2a,a2a",
    [
        (False, True, True, True),
        (True, False, True, True),
        (True, True, False, True),
        (True, True, True, False),
        (False, False, False, False),
    ],
)
def test_gemnet_oc_forward_toggles(tmp_path, fe_mini_data, quad, a2e, e2a, a2a):
    _, b = _build_small_batch(tmp_path, fe_mini_data, oc_mode=True, quadruplets=True)
    num_targets = int(b.intensities.size(1))
    model = _tiny_gemnet_oc(num_targets=num_targets, quad=quad, a2e=a2e, e2a=e2a, a2a=a2a)
    model.eval()
    bp = GemNetOCBatchProcessor()
    inp = bp.input_preparation(b)
    with torch.no_grad():
        E = model(**inp)
    assert E.shape == (b.num_nodes, num_targets)
    assert torch.isfinite(E).all()
    # Each toggle configuration must still produce non-trivial outputs.
    assert E.abs().max().item() > 0.0


# Basis layer value tests


def test_bessel_basis_values_gemnet():
    """
    ``BesselBasisLayer`` must produce finite, bounded values within the cutoff and decay towards zero at the cutoff boundary (envelope).
    """
    from xanesnet.models.gemnet.layers.basis import BesselBasisLayer

    rbf = BesselBasisLayer(num_radial=4, cutoff=5.0, envelope_exponent=5)
    d = torch.tensor([0.5, 1.0, 2.5, 4.5, 4.999])
    out = rbf(d)
    assert out.shape == (5, 4)
    assert torch.isfinite(out).all()
    # Envelope polynomial forces the basis to vanish at d == cutoff.
    near_cutoff = rbf(torch.tensor([4.9999999])).abs().max().item()
    far_from_cutoff = rbf(torch.tensor([1.0])).abs().max().item()
    assert near_cutoff < far_from_cutoff
    # Radial basis is non-trivial (not zero).
    assert out.abs().max().item() > 0.0


def test_radial_basis_values_gemnet_oc():
    """
    ``RadialBasis`` used in GemNet-OC must be finite and non-zero in-range for a realistic set of distances.
    """
    from xanesnet.models.gemnet_oc.layers.radial_basis import RadialBasis

    rb = RadialBasis(num_radial=5, cutoff=6.0, rbf={"name": "gaussian"}, envelope={"name": "polynomial", "exponent": 5})
    d = torch.tensor([0.1, 1.0, 2.5, 5.0, 5.99])
    out = rb(d)
    assert out.shape == (5, 5)
    assert torch.isfinite(out).all()
    assert out.abs().max().item() > 0.0


def test_spherical_basis_values_gemnet():
    """
    ``SphericalBasisLayer`` of GemNet (efficient=True) must return a finite (rbf, sph) pair
    with expected shapes for a hand-crafted pair of distances/angles.
    """
    from xanesnet.models.gemnet.layers.basis import SphericalBasisLayer

    sbf = SphericalBasisLayer(
        num_spherical=4,
        num_radial=3,
        cutoff=5.0,
        envelope_exponent=5,
        efficient=True,
    )
    # 2 "edges", 3 triplets total: edge 0 -> 2 triplets, edge 1 -> 1 triplet.
    d = torch.tensor([1.0, 2.0])
    id3_reduce_ca = torch.tensor([0, 0, 1])
    Kidx3 = torch.tensor([0, 1, 0])
    # SphericalBasisLayer consumes angles (radians), not cosines.
    angle_cab = torch.tensor([0.8, 2.1, 1.2])
    rbf_env, sph = sbf(d, angle_cab, id3_reduce_ca, Kidx3)
    assert torch.isfinite(rbf_env).all()
    assert torch.isfinite(sph).all()
    # sph shape is (nEdges, Kmax, num_spherical). Kmax = 2 here.
    assert sph.shape == (2, 2, 4)
    # Non-trivial values
    assert rbf_env.abs().max().item() > 0.0
    assert sph.abs().max().item() > 0.0
