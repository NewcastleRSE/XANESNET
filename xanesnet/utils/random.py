import random

import numpy as np
import torch


def set_global_seed(seed: int | None = None) -> int:
    """
    Set the random seed for Python, NumPy, and PyTorch (CPU + CUDA).
    """
    if seed is None:
        seed = random.randint(0, 2**32 - 1)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    return seed
