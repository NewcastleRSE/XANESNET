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

# Ported from fairchem reference (MIT). Pure utility code.

from __future__ import annotations

from functools import partial

import torch


def _standardize(kernel: torch.Tensor) -> torch.Tensor:
    eps = 1e-6
    axis = [0, 1] if len(kernel.shape) == 3 else 1
    var, mean = torch.var_mean(kernel, dim=axis, unbiased=True, keepdim=True)
    return (kernel - mean) / (var + eps) ** 0.5


def he_orthogonal_init(tensor: torch.Tensor) -> torch.Tensor:
    """He-orthogonal initialisation (Saxe / He / Kaiming hybrid)."""
    tensor = torch.nn.init.orthogonal_(tensor)
    fan_in = tensor.shape[:-1].numel() if len(tensor.shape) == 3 else tensor.shape[1]
    with torch.no_grad():
        tensor.data = _standardize(tensor.data)
        tensor.data *= (1 / fan_in) ** 0.5
    return tensor


def grid_init(tensor: torch.Tensor, start: int = -1, end: int = 1) -> torch.Tensor:
    fan_in = tensor.shape[1]
    with torch.no_grad():
        data = torch.linspace(start, end, fan_in, device=tensor.device, dtype=tensor.dtype).expand_as(tensor)
        tensor.copy_(data)
    return tensor


def log_grid_init(tensor: torch.Tensor, start: int = -4, end: int = 0) -> torch.Tensor:
    fan_in = tensor.shape[1]
    with torch.no_grad():
        data = torch.logspace(start, end, fan_in, device=tensor.device, dtype=tensor.dtype).expand_as(tensor)
        tensor.copy_(data)
    return tensor


def get_initializer(name: str, **init_kwargs):
    name = name.lower()
    if name == "heorthogonal":
        initializer = he_orthogonal_init
    elif name == "zeros":
        initializer = torch.nn.init.zeros_
    elif name == "grid":
        initializer = grid_init
    elif name == "loggrid":
        initializer = log_grid_init
    else:
        raise ValueError(f"Unknown initializer: {name}")
    return partial(initializer, **init_kwargs)
