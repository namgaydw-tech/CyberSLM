from __future__ import annotations

import os
import random

import torch


def set_seed(seed: int, *, deterministic: bool = True) -> None:
    """Seed Python and PyTorch. PYTHONHASHSEED affects child processes only."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
