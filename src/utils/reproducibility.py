"""Determinism flags. Per §9 of the plan, single-GPU FP32 should be bit-exact;
multi-GPU bf16 cannot be made fully deterministic and is documented as such.
"""
from __future__ import annotations

import os
import random


def set_seed(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
