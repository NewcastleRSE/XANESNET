# SPDX-License-Identifier: GPL-3.0-or-later
#
# XANESNET
#
# Authors:  Hendrik Junkawitsch, Tom J. Penfold, Thomas J. Pope, C. D. Rankine, B. Li
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
#
# Citations:
#   Junkawitsch et al., "XANESNET: A Modular, Extensible, and Flexible Machine Learning Framework for Spectroscopy."
#   Rankine et al., "A Deep Neural Network for the Rapid Prediction of X-ray Absorption Spectra."
#   Rankine et al., "Accurate, affordable, and generalizable machine learning simulations of transition metal x-ray absorption spectra using the XANESNET deep neural network."
#   Penfold et al., "Field-Aware Energy-Conditioned Message Passing Neural Networks for Absorber-Centred Modelling of X-ray Spectroscopy."
#   Penfold et al., "A deep neural network for valence-to-core X-ray emission spectroscopy."
#   Falbo et al., "On the Analysis of X-ray Absorption Spectra for Polyoxometallates."
#   Madkhali et al., "Enhancing the Analysis of Disorder in X-ray Absorption Spectra: Application of Deep Neural Networks to T-Jump X-ray Probe Experiments."
#   Madkhali et al., "The Role of Structural Representation in the Performance of a Deep Neural Network for X-ray Spectroscopy."

"""Unit tests for the shared spectral math helpers (symmetric FFT and Gaussian basis)."""

import pytest
import torch

from xanesnet.utils.math import (
    SpectralBasis,
    SpectralPost,
    build_ridge_operator,
    fft,
    gaussian_fit,
    gaussian_inverse,
    inverse_fft,
)


@pytest.fixture
def signal() -> torch.Tensor:
    generator = torch.Generator().manual_seed(0)
    return torch.rand(3, 16, generator=generator)


@pytest.fixture
def basis() -> SpectralBasis:
    energies = torch.linspace(0.0, 10.0, 40)
    return SpectralBasis(energies, widths_eV=[0.5, 1.0], stride=2)


###############################################################################
#################################### FFT ######################################
###############################################################################


@pytest.mark.parametrize("concat, factor", [(False, 2), (True, 3)])
def test_fft_output_length(signal, concat, factor):
    assert fft(signal, concat=concat).shape == (3, signal.shape[-1] * factor)


def test_fft_with_concat_prepends_the_original_signal(signal):
    out = fft(signal, concat=True)
    torch.testing.assert_close(out[..., : signal.shape[-1]], signal)
    torch.testing.assert_close(out[..., signal.shape[-1] :], fft(signal, concat=False))


@pytest.mark.parametrize("concat", [False, True])
def test_inverse_fft_recovers_the_signal(signal, concat):
    restored = inverse_fft(fft(signal, concat=concat), concat=concat)
    torch.testing.assert_close(restored, signal, atol=1e-4, rtol=1e-4)


def test_fft_roundtrip_handles_unbatched_input():
    x = torch.linspace(0, 1, 8)
    torch.testing.assert_close(inverse_fft(fft(x, concat=False), concat=False), x, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("concat, length", [(True, 10), (False, 9)])
def test_inverse_fft_rejects_incompatible_lengths(concat, length):
    with pytest.raises(ValueError, match="divisible"):
        inverse_fft(torch.zeros(length), concat=concat)


###############################################################################
############################### GAUSSIAN BASIS ################################
###############################################################################


def test_basis_dimensions_follow_widths_and_stride(basis):
    n_points, n_basis = 40, 2 * 20
    assert basis.Phi.shape == (n_points, n_basis)
    assert basis.centers.shape == (n_basis,)


def test_basis_columns_are_normalized_by_default(basis):
    dE = float(basis.E[1] - basis.E[0])
    torch.testing.assert_close(basis.Phi.sum(dim=0) * dE, torch.ones(basis.Phi.shape[1]), atol=1e-4, rtol=1e-4)


def test_unnormalized_basis_peaks_at_one():
    energies = torch.linspace(0.0, 10.0, 40)
    plain = SpectralBasis(energies, widths_eV=[1.0], normalize_atoms=False)
    assert plain.Phi.max() == pytest.approx(1.0)


def test_basis_buffers_are_not_trainable(basis):
    assert list(basis.parameters()) == []


def test_synthesize_is_linear_in_the_coefficients(basis):
    coeffs = torch.rand(4, basis.Phi.shape[1])
    torch.testing.assert_close(basis.synthesize(coeffs), coeffs @ basis.Phi.T)


def test_device_copy_on_cpu_returns_the_same_instance(basis):
    assert basis.device_copy(torch.device("cpu")) is basis


def test_ridge_operator_shape_and_dtype(basis):
    operator = build_ridge_operator(basis.Phi)
    assert operator.shape == (basis.Phi.shape[1], basis.Phi.shape[0])
    assert operator.dtype == torch.float32


def test_ridge_operator_handles_a_singular_gram_matrix():
    phi = torch.ones(5, 3)  # rank one, so the un-regularized Gram matrix is singular
    operator = build_ridge_operator(phi, lam=1e-6)
    assert torch.isfinite(operator).all()


def test_gaussian_fit_and_inverse_reconstruct_a_smooth_spectrum(basis):
    energies = basis.E
    spectrum = torch.exp(-0.5 * ((energies - 5.0) / 1.0) ** 2).unsqueeze(0)

    reconstructed = gaussian_inverse(basis, gaussian_fit(basis, spectrum))

    assert reconstructed.shape == spectrum.shape
    assert torch.nn.functional.mse_loss(reconstructed, spectrum) < 1e-3


def test_gaussian_inverse_can_clamp_negative_values(basis):
    coeffs = -torch.ones(1, basis.Phi.shape[1])
    assert gaussian_inverse(basis, coeffs).min() < 0
    assert gaussian_inverse(basis, coeffs.clone(), nonneg_output=True).min() >= 0


def test_spectral_post_matches_basis_synthesis_and_has_no_parameters(basis):
    post = SpectralPost(basis)
    coeffs = torch.rand(2, basis.Phi.shape[1])

    torch.testing.assert_close(post(coeffs), basis.synthesize(coeffs))
    torch.testing.assert_close(post.forward_from_coeffs(coeffs), post(coeffs))
    assert list(post.parameters()) == []


def test_spectral_post_nonneg_option_clamps(basis):
    post = SpectralPost(basis, nonneg_output=True)
    assert post(-torch.ones(1, basis.Phi.shape[1])).min() >= 0
