"""
Contains numba-optimized functions for when the kernel is set as a tuple.
"""
from __future__ import annotations

# IMPORTs alias
import numpy as np

# IMPORTs sub
from numba import njit, prange

# API public
__all__ = ['tuple_sliding_nanmedian_3d', 'tuple_sliding_nanmean_3d']



@njit
def _fast_median(window: np.ndarray) -> np.floating | float:
    """
    To get the median of an odd sized kernel using partitioning.
    Also takes into account NaN values.

    Args:
        window (np.ndarray): the window to get the median value.

    Returns:
        np.floating | float: the median value of the window. If the window is empty, returns
            NaN.
    """

    valid = window[~np.isnan(window)]
    n = valid.size
    if n == 0: return np.nan

    # ODD
    if n % 2 == 1: return np.partition(valid, n // 2)[n // 2]

    # EVEN
    partitioned = np.partition(valid, [n // 2 - 1, n // 2])
    return 0.5 * (partitioned[n // 2 - 1] + partitioned[n // 2])

@njit(parallel=True)
def tuple_sliding_nanmedian_3d(data: np.ndarray, kernel: tuple[int, int, int]) -> np.ndarray:
    """
    To get the sliding median value for a given kernel not containing any weights. Keep in mind
    that the input data must be pre-padded to handle borders correctly.
    This is done using numba as I didn't find any other way to efficiently get the sliding
    median while there are NaN values inside the data.

    Args:
        data (np.ndarray): the padded data to get the sliding median for. Can and should contain
            NaNs.
        kernel (tuple[int, int, int]): the shape of the kernel.

    Returns:
        np.ndarray: the sliding median result.
    """

    depth, rows, cols = data.shape
    pad_d, pad_r, pad_c = kernel[0] // 2, kernel[1] // 2, kernel[2] // 2
    results = np.empty((depth - 2 * pad_d, rows - 2 * pad_r, cols - 2 * pad_c), dtype=data.dtype)

    for j in prange(cols - 2 * pad_c):  # as cols is 128 but depth is 36 and rows 1024
        for i in range(rows - 2 * pad_r):
            for d in range(depth - 2 * pad_d):
                window = data[d:d + kernel[0], i:i + kernel[1], j:j + kernel[2]].ravel()
                results[d, i, j] = _fast_median(window)
    return results

@njit(parallel=True)
def tuple_sliding_nanmean_3d(data: np.ndarray, kernel: tuple[int, int, int]) -> np.ndarray:
    """
    To get the sliding mean value for a given kernel not containing any weights. Keep in mind that
    the input data must be pre-padded to handle borders correctly.

    Args:
        data (np.ndarray): the padded data to get the sliding mean for. Can and should contain
            NaNs.
        kernel (tuple[int, int, int]): the shape of the kernel.

    Returns:
        np.ndarray: the sliding mean result.
    """

    depth, rows, cols = data.shape
    pad_d, pad_r, pad_c = kernel[0] // 2, kernel[1] // 2, kernel[2] // 2
    results = np.empty((depth - 2 * pad_d, rows - 2 * pad_r, cols - 2 * pad_c), dtype=data.dtype)

    for j in prange(cols - 2 * pad_c):  # as cols is 128 but depth is 36 and rows 1024
        for i in range(rows - 2 * pad_r):
            for d in range(depth - 2 * pad_d):
                window = data[d:d + kernel[0], i:i + kernel[1], j:j + kernel[2]].ravel()

                valid = window[~np.isnan(window)]
                n = valid.size
                if n == 0:
                    results[d, i, j] = np.nan
                else:
                    results[d, i, j] = np.mean(valid)
    return results
