"""Reusable play isolation metrics."""

from __future__ import annotations

import numpy as np


def mean_absolute_isolation(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    result = np.full(array.shape, np.nan, dtype=float)
    valid = np.isfinite(array)
    if valid.sum() < 2:
        return result
    selected = array[valid]
    distances = np.abs(selected[:, None] - selected[None, :])
    np.fill_diagonal(distances, np.nan)
    result[valid] = np.nanmean(distances, axis=1)
    return result
