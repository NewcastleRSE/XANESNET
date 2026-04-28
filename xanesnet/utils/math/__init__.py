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

"""Public API for XANESNET mathematical utilities."""

from .fourier import fft, inverse_fft
from .gaussian import (
    SpectralBasis,
    SpectralPost,
    build_ridge_operator,
    gaussian_fit,
    gaussian_inverse,
)

__all__ = [
    "fft",
    "inverse_fft",
    "SpectralBasis",
    "SpectralPost",
    "build_ridge_operator",
    "gaussian_fit",
    "gaussian_inverse",
]
