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

import torch


def fft(x: torch.Tensor, concat: bool = False) -> torch.Tensor:
    x = x.to(dtype=torch.float32)
    x_rev = torch.flip(x, dims=[-1])
    y = torch.cat([x, x_rev], dim=-1)
    f = torch.fft.fft(y)
    z = f.real

    if concat:
        z = torch.cat([x, z], dim=-1)

    return z


def inverse_fft(z: torch.Tensor, concat: bool = False) -> torch.Tensor:
    z = z.to(dtype=torch.float32)
    L = z.shape[-1]

    if concat:
        N = L // 3
        z = z[..., N:]
    else:
        N = L // 2

    iz = torch.fft.ifft(z, dim=-1).real
    iz = iz[..., :N]

    return iz
