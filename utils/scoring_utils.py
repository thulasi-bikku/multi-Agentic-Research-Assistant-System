from __future__ import annotations

import numpy as np


def normalize_array(values: list[float] | np.ndarray) -> np.ndarray:
    arr = np.array(values, dtype=float)
    if arr.size == 0:
        return arr

    max_abs = np.max(np.abs(arr))
    if max_abs == 0:
        return np.zeros_like(arr)

    return arr / max_abs
