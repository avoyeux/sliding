"""
Codes to compute the sliding mean of a 3D array.
Done here so that I can compare these results, which should be correct, to the convolution-based
implementation (to check the conversion between the different border types).
"""
from __future__ import annotations

# IMPORTs alias
import numpy as np

# IMPORTs sub
from numba import njit, prange

# API public
__all__ = ['sliding_weighted_mean_3d', 'tuple_sliding_nanmean_3d']



@njit
def _kernel_setup(
        kernel: np.ndarray[tuple[int, int, int], np.dtype[np.float64]],
    ) -> tuple[
        np.ndarray[tuple[int], np.dtype[np.float64]],
        np.ndarray[tuple[int], np.dtype[np.bool_]],
    ]:
    """
    To flip the kernel (correlation vs convolution) and get the needed kernel mask.

    Args:
        kernel (np.ndarray[tuple[int, int, int], np.dtype[np.float64]]): the kernel to setup.

    Returns:
        tuple[
            np.ndarray[tuple[int], np.dtype[np.float64]],
            np.ndarray[tuple[int], np.dtype[np.bool_]],
        ]: the flattened flipped kernel and the mask of valid weights.
    """

    kernel_flip = np.ascontiguousarray(kernel[::-1, ::-1, ::-1])
    flat_kernel = kernel_flip.ravel()
    valid_kernel_mask = (np.abs(flat_kernel) >= 1e-10)
    return flat_kernel, valid_kernel_mask#type:ignore

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
    flat_kernel, kernel_mask = _kernel_setup(kernel)

    for j in prange(cols - 2 * pad_c):  # as cols is 128 but depth is 36 and rows 1024
        for i in range(rows - 2 * pad_r):
            for d in range(depth - 2 * pad_d):
                window = data[d:d + kd, i:i + kr, j:j + kc].ravel()
                valid_values, valid_weights = _apply_kernel_weights(
                    flat_window=window,
                    flat_kernel=flat_kernel,
                    flat_kernel_not_nan=kernel_mask,
                )

                if valid_values.size == 0:
                    results[d, i, j] = np.nan
                else:
                    results[d, i, j] = np.sum(valid_values * valid_weights) / np.sum(valid_weights)
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
