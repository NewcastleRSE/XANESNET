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
Regression tests that ``Dense``, ``ResidualLayer`` and ``OutputBlock`` apply
their custom He-orthogonal / configured output initialization inside
``__init__`` (i.e. without requiring an explicit ``init_weights`` call from
the outside).

Motivation: the reference GemNet implementation calls ``reset_parameters`` at
the end of every layer's ``__init__``. Forgetting this in the port silently
leaves torch's default ``kaiming_uniform_`` weights in place, producing
slightly different (and untested) numerics from the reference. These tests
catch any future regression of that behaviour.
"""

import torch

from xanesnet.models.gemnet.layers.atom_update import OutputBlock
from xanesnet.models.gemnet.layers.base import Dense, ResidualLayer


def _standardized(w: torch.Tensor) -> torch.Tensor:
    """Return ``w`` standardized the way ``he_orthogonal_init`` does along the
    input axis (axis=1 for 2D tensors)."""
    var, mean = torch.var_mean(w, dim=1, unbiased=True, keepdim=True)
    return (w - mean) / (var + 1e-6) ** 0.5


def test_dense_init_is_he_orthogonal_without_reset() -> None:
    torch.manual_seed(0)
    in_f, out_f = 32, 24
    layer = Dense(in_f, out_f, bias=False, activation="swish")

    w = layer.linear.weight.detach()
    assert w.shape == (out_f, in_f)

    # Per-row (output-axis) mean ~0 and var ~1 after ``_standardize``,
    # i.e. the input-axis has been standardized inside he_orthogonal_init.
    ws = _standardized(w)
    # Rows should be (approximately) orthonormal up to the He variance scaling.
    # Since he_orthogonal_init multiplies by 1/sqrt(fan_in), the raw std is ~1/sqrt(in_f).
    std = w.std().item()
    assert 0.5 / in_f**0.5 < std < 2.0 / in_f**0.5, (
        f"Dense weight std {std:.4f} inconsistent with He-orthogonal init (expected ~{1/in_f**0.5:.4f}); "
        "did reset_parameters() run inside __init__?"
    )
    # Standardization should leave ws with ~zero mean per row.
    assert ws.mean(dim=1).abs().max().item() < 1e-4


def test_residual_layer_init_is_he_orthogonal_without_reset() -> None:
    torch.manual_seed(0)
    units = 16
    layer = ResidualLayer(units, activation="swish", nLayers=2)
    for dense in layer.dense_mlp:
        std = dense.linear.weight.std().item()
        assert 0.5 / units**0.5 < std < 2.0 / units**0.5


def test_output_block_init_respects_output_init_zeros() -> None:
    """``output_init='zeros'`` should zero ``out_energy`` even without an
    explicit ``init_weights`` call."""
    block = OutputBlock(
        emb_size_atom=8,
        emb_size_edge=8,
        emb_size_rbf=4,
        nHidden=1,
        num_targets=3,
        activation="swish",
        output_init="zeros",
        scale_file=None,
        name="OutBlock_test",
    )
    assert torch.all(block.out_energy.weight == 0.0)


def test_output_block_init_respects_output_init_heorthogonal() -> None:
    block = OutputBlock(
        emb_size_atom=8,
        emb_size_edge=8,
        emb_size_rbf=4,
        nHidden=1,
        num_targets=3,
        activation="swish",
        output_init="HeOrthogonal",
        scale_file=None,
        name="OutBlock_test",
    )
    w = block.out_energy.weight.detach()
    # With he_orthogonal_init, per-input-axis standardization produces std ~ 1/sqrt(fan_in).
    fan_in = w.shape[1]
    std = w.std().item()
    assert 0.5 / fan_in**0.5 < std < 2.0 / fan_in**0.5, (
        f"out_energy std {std:.4f} inconsistent with He-orthogonal init; " "did reset_parameters() run inside __init__?"
    )
