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


def fft(x, concat):
    """
    Transform xanes spectra using Fourier
    """

    x = np.asarray(x)
    y = np.hstack((x, x[::-1]))
    f = np.fft.fft(y)
    z = f.real

    # Combine features
    if concat:
        z = np.concatenate((x, z))

    return z


def inverse_fft(z, concat):
    """
    Get inverse of fourier transformed data
    """
    # Decompose features
    if concat:
        z = z[z.size // 3 :]

    iz = np.fft.ifft(z).real[: z.size // 2]

    return iz
