"""Deterministic seeding across python / numpy / torch (CUDA & ROCm)."""
from __future__ import annotations

import os
import random
from typing import Optional

import numpy as np


def seed_everything(seed: int, deterministic_torch: bool = True) -> int:
    """Seed python, numpy and (if installed) torch RNGs.

    Returns the seed so callers can log it. ``deterministic_torch`` trades a
    little throughput for bit-reproducibility; turn it off for max speed on the
    MI300X once results are stable.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():           # True on ROCm builds too
            torch.cuda.manual_seed_all(seed)
        if deterministic_torch:
            # cudnn flags are honored by the ROCm/MIOpen backend as well.
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            # Opt-in deterministic algorithms; warn_only avoids hard crashes on
            # ops without a deterministic implementation.
            try:
                torch.use_deterministic_algorithms(True, warn_only=True)
            except TypeError:                    # older torch
                torch.use_deterministic_algorithms(True)
    except ImportError:
        pass
    return seed


def worker_init_fn(worker_id: int, base_seed: Optional[int] = None) -> None:
    """DataLoader worker seeding so augmentations differ but stay reproducible."""
    try:
        import torch

        seed = (torch.initial_seed() if base_seed is None else base_seed) % (2**32)
    except ImportError:
        seed = (base_seed or 0)
    seed = (seed + worker_id) % (2**32)
    np.random.seed(seed)
    random.seed(seed)
