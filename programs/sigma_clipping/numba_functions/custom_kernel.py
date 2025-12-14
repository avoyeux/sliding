"""
Contains numba-optimized functions for a custom numpy array kernel in the sigma clipping.
"""
from __future__ import annotations

# IMPORTs alias
import numpy as np

# IMPORTs sub
from numba import njit, prange

# API public
__all__ = ['sliding_weighted_median_3d', 'sliding_weighted_mean_3d']



@njit
def _apply_kernel_weights(
        flat_window: np.ndarray,
        flat_kernel: np.ndarray[tuple[int], np.dtype[np.float64]],
        flat_kernel_not_nan: np.ndarray[tuple[int], np.dtype[np.bool_]],
    ) -> tuple[np.ndarray, np.ndarray]:
    """
    Gives the valid values and weights for a given window.

    Args:
        flat_window (np.ndarray): the flatten window to apply the kernel on.
        flat_kernel (np.ndarray[tuple[int], np.dtype[np.float64]]): the flattened kernel.
        flat_kernel_not_nan (np.ndarray[tuple[int], np.dtype[np.bool_]]): the mask of the flattened
            kernel indicating which weights are valid (not NaN and not zero).

    Returns:
        tuple[np.ndarray, np.ndarray]: the valid values and weights.
    """

    # VALUEs valid
    valid_mask = ~np.isnan(flat_window) & flat_kernel_not_nan
    valid_values = flat_window[valid_mask]
    valid_weights = flat_kernel[valid_mask]
    return valid_values, valid_weights

@njit
def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    """
    Gives the statistical weighted median value.

    Args:
        values (np.ndarray): the values for which to find the weighted median.
        weights (np.ndarray): the weights for the values.

    Returns:
        float: the statistical weighted median.
    """

    # SORT together
    sort_idx = np.argsort(values)
    sorted_values = values[sort_idx]
    sorted_weights = weights[sort_idx]

    # SUM weights
    cumsum = np.cumsum(sorted_weights)
    total_weight = cumsum[-1]

    # STATISTICAL MEDIAN - 50% of total weight
    median_weight = total_weight / 2.

    # INDEX
    idx = np.searchsorted(cumsum, median_weight)
    if idx >= len(sorted_values): idx = len(sorted_values) - 1
    return sorted_values[idx]

@njit(parallel=True)
def sliding_weighted_median_3d(
        data: np.ndarray,
        kernel: np.ndarray[tuple[int, int, int], np.dtype[np.float64]],
    ) -> np.ndarray:
    """
    To get the sliding median value for a given weighted kernel. Keep in mind that the input data
    must be pre-padded to handle borders correctly.

    Args:
        data (np.ndarray): the padded data to get the sliding median for. Can and should contain
            NaNs.
        kernel (np.ndarray[tuple[int, int, int], np.dtype[np.float64]]): the weighted kernel.

    Returns:
        np.ndarray: the sliding median result.
    """

    depth, rows, cols = data.shape
    kd, kr, kc = kernel.shape
    pad_d, pad_r, pad_c = kd // 2, kr // 2, kc // 2
    results = np.empty((depth - 2 * pad_d, rows - 2 * pad_r, cols - 2 * pad_c), dtype=data.dtype)

    # KERNEL setup
    flat_kernel = kernel.ravel()
    flat_kernel_not_nan = ~np.isnan(flat_kernel) & (np.abs(flat_kernel) >= 1e-10)

    for j in prange(cols - 2 * pad_c):  # as cols is 128 but depth is 36 and rows 1024
        for i in range(rows - 2 * pad_r):
            for d in range(depth - 2 * pad_d):
                window = data[d:d + kd, i:i + kr, j:j + kc].ravel()
                valid_values, valid_weights = _apply_kernel_weights(
                    flat_window=window,
                    flat_kernel=flat_kernel,
                    flat_kernel_not_nan=flat_kernel_not_nan,#type:ignore
                )

                if valid_values.size == 0:
                    results[d, i, j] = np.nan
                else:
                    results[d, i, j] = _weighted_median(valid_values, valid_weights)
    return results

@njit(parallel=True)
def sliding_weighted_mean_3d(
        data: np.ndarray,
        kernel: np.ndarray[tuple[int, int, int], np.dtype[np.float64]],
    ) -> np.ndarray:
    """
    To get the sliding mean value for a given weighted kernel. Keep in mind that the input data
    must be pre-padded to handle borders correctly.

    Args:
        data (np.ndarray): the padded data to get the sliding mean for. Can and should contain
            NaNs.
        kernel (np.ndarray[tuple[int, int, int], np.dtype[np.float64]]): the weighted kernel.

    Returns:
        np.ndarray: the sliding mean result.
    """

    depth, rows, cols = data.shape
    kd, kr, kc = kernel.shape
    pad_d, pad_r, pad_c = kd // 2, kr // 2, kc // 2
    results = np.empty((depth - 2 * pad_d, rows - 2 * pad_r, cols - 2 * pad_c), dtype=data.dtype)

    # KERNEL setup
    flat_kernel = kernel.ravel()
    flat_kernel_not_nan = ~np.isnan(flat_kernel) & (np.abs(flat_kernel) >= 1e-10)

    for j in prange(cols - 2 * pad_c):  # as cols is 128 but depth is 36 and rows 1024
        for i in range(rows - 2 * pad_r):
            for d in range(depth - 2 * pad_d):
                window = data[d:d + kd, i:i + kr, j:j + kc].ravel()
                valid_values, valid_weights = _apply_kernel_weights(
                    flat_window=window,
                    flat_kernel=flat_kernel,
                    flat_kernel_not_nan=flat_kernel_not_nan,#type:ignore
                )

                if valid_values.size == 0:
                    results[d, i, j] = np.nan
                else:
                    results[d, i, j] = np.sum(valid_values * valid_weights) / np.sum(valid_weights)
    return results
