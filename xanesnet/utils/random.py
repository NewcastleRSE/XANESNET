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

"""Random seed management for reproducible XANESNET experiments."""

import random
from numbers import Integral

import numpy as np
import torch


def set_global_seed(seed: int | None = None) -> int:
    """Set the random seed for Python, NumPy, and PyTorch (CPU + CUDA).

    Args:
        seed: Seed value to use. If ``None``, a random seed is drawn from
            the system RNG. Explicit seeds must lie in
            ``[0, 2**32 - 1]``.

    Returns:
        The seed that was actually applied.

    Raises:
        TypeError: If ``seed`` is not an integer or ``None``.
        ValueError: If ``seed`` is outside ``[0, 2**32 - 1]``.
    """
    max_seed = 2**32 - 1

    if seed is None:
        seed = random.SystemRandom().randrange(max_seed + 1)
    else:
        if not isinstance(seed, Integral):
            raise TypeError(f"Seed must be an integer or None, got {type(seed).__name__}.")
        seed = int(seed)
        if not 0 <= seed <= max_seed:
            raise ValueError(f"Seed must be between 0 and {max_seed}")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    return seed
