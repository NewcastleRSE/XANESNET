import random
from typing import Optional

import numpy as np
import torch


def set_global_seed(seed: Optional[int] = None) -> int:
    """
    Set the random seed for Python, NumPy, and PyTorch (CPU + CUDA).

    Args:
        seed (int, optional): The seed to use. If None, a random 32-bit seed is generated.

    Returns:
        int: The seed that was set.
    """
    if seed is None:
        seed = random.randint(0, 2**32 - 1)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    return seed
