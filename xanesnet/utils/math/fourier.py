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

"""Symmetric-extension FFT transforms for XANESNET spectral processing."""

import math

import torch


def _reconstruct_symmetric_spectrum(z: torch.Tensor) -> torch.Tensor:
    """Reconstruct the complex FFT spectrum implied by the mirrored signal.

    For ``y = [x, flip(x)]``, the FFT has a deterministic phase offset of
    ``exp(i * pi * k / L)`` relative to a real cosine transform. ``fft``
    stores only the real part, which is still sufficient to recover the full
    spectrum because the mirrored construction fixes that phase.

    Args:
        z: ``(..., L)`` float -- real-valued spectrum returned by ``fft``
            after optional concatenation has been stripped.

    Returns:
        ``(..., L)`` complex -- full FFT spectrum compatible with
        ``torch.fft.ifft``.
    """
    length = z.shape[-1]
    if length == 0:
        return z.to(dtype=torch.complex64)

    angle = math.pi * torch.arange(length, device=z.device, dtype=z.dtype) / length
    cos_angle = torch.cos(angle)
    phase = torch.polar(torch.ones_like(angle), angle)

    spectrum = torch.zeros(z.shape, device=z.device, dtype=torch.complex64)
    nonzero = cos_angle.abs() > 1e-7
    spectrum[..., nonzero] = (z[..., nonzero] / cos_angle[nonzero]).to(torch.complex64) * phase[nonzero]
    # The quarter-turn bin has zero real component for mirrored inputs.
    spectrum[..., ~nonzero] = 0
    return spectrum


def fft(x: torch.Tensor, concat: bool) -> torch.Tensor:
    """Apply a symmetric-extension FFT transform.

    Constructs an even-symmetric sequence ``[x, flip(x)]`` of length ``2N``
    and takes the real part of its FFT. This is analogous to a DCT-II
    transform.

    Args:
        x: Input tensor of any batch shape; the last dimension is transformed.
        concat: If ``True``, prepend the original ``x`` to the output,
            producing shape ``(..., 3N)``. If ``False``, return only the
            ``(..., 2N)`` FFT output.

    Returns:
        Transformed tensor of shape ``(..., 3N)`` when ``concat=True``,
        or ``(..., 2N)`` when ``concat=False``, where ``N = x.shape[-1]``.
    """
    x = x.to(dtype=torch.float32)
    x_rev = torch.flip(x, dims=[-1])
    y = torch.cat([x, x_rev], dim=-1)
    f = torch.fft.fft(y)
    z = f.real

    if concat:
        z = torch.cat([x, z], dim=-1)

    return z


def inverse_fft(z: torch.Tensor, concat: bool) -> torch.Tensor:
    """Invert the symmetric-extension FFT transform applied by ``fft``.

    Args:
        z: Transformed tensor. Shape ``(..., 3N)`` when the corresponding
            ``fft`` call used ``concat=True``, or ``(..., 2N)`` otherwise.
        concat: Must match the ``concat`` flag used in the corresponding
            ``fft`` call. If ``True``, the leading ``N`` elements (the
            original signal) are stripped before inverting.

    Returns:
        Reconstructed signal of shape ``(..., N)``.

    Raises:
        ValueError: If the last dimension is incompatible with ``concat``.
    """
    z = z.to(dtype=torch.float32)
    L = z.shape[-1]

    if concat:
        if L % 3 != 0:
            raise ValueError(f"Expected a last dimension divisible by 3 when concat=True, got {L}.")
        N = L // 3
        z = z[..., N:]
    else:
        if L % 2 != 0:
            raise ValueError(f"Expected a last dimension divisible by 2 when concat=False, got {L}.")
        N = L // 2

    spectrum = _reconstruct_symmetric_spectrum(z)
    iz = torch.fft.ifft(spectrum, dim=-1).real
    iz = iz[..., :N]

    return iz
