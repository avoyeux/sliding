"""
Contains numba-optimized functions to compute the sliding median of n-dimensional arrays given
a kernel, with or without weights.
"""
from __future__ import annotations

# IMPORTs alias
import numpy as np

# IMPORTs sub
from numba import njit, prange

# TYPE ANNOTATIONs
# from typing import cast

# API public
__all__ = [
    'tuple_sliding_nanmedian_3d',
    'sliding_weighted_median_3d',
    'tuple_sliding_nanmedian_nd',
    'sliding_weighted_median_nd',
]

# ! 3-D implementation for kernel with weights seems slower than n-D implementation



@njit(parallel=True)
def tuple_sliding_nanmedian_3d[Data: np.ndarray](data: Data, kernel: tuple[int, int, int]) -> Data:
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

    for i in prange(rows - 2 * pad_r):  # over rows (1024) instead of cols (128) or depth (36)
        for j in range(cols - 2 * pad_c):
            for d in range(depth - 2 * pad_d):
                window = data[d:d + kernel[0], i:i + kernel[1], j:j + kernel[2]].ravel()
                results[d, i, j] = _fast_median(window)
    return results#type:ignore

def tuple_sliding_nanmedian_nd[Data: np.ndarray](data: Data, kernel: tuple[int, ...]) -> Data:
    """
    To compute the sliding median for n-dimensional data and kernel.

    Args:
        data (np.ndarray): the n-dimensional data for which to get the sliding median.
        kernel (tuple[int, ...]): the shape of the kernel (has to have the same dimensionality as
            data).

    Returns:
        np.ndarray: the sliding median results.
    """

    # SHAPE output
    output_shape_tuple = tuple(int(d - k + 1) for d, k in zip(data.shape, kernel))
    output_shape_arr = np.array(output_shape_tuple, dtype=np.int64)

    # MEDIAN sliding
    result_flat = _tuple_sliding_nanmedian_nd(data, kernel, output_shape_arr)
    return result_flat.reshape(output_shape_tuple)#type:ignore

@njit(parallel=True)
def sliding_weighted_median_3d[Data: np.ndarray](
        data: Data,
        kernel: np.ndarray[tuple[int, int, int], np.dtype[np.floating]],
    ) -> Data:
    """
    To get the sliding median value for a given weighted kernel. Keep in mind that the input data
    must be pre-padded to handle borders correctly.

    Args:
        data (np.ndarray): the padded data to get the sliding median for. Can and should contain
            NaNs.
        kernel (np.ndarray[tuple[int, int, int], np.dtype[np.floating]]): the weighted kernel.

    Returns:
        np.ndarray: the sliding median result.
    """

    depth, rows, cols = data.shape
    kd, kr, kc = kernel.shape
    pad_d, pad_r, pad_c = kd // 2, kr // 2, kc // 2
    results = np.empty((depth - 2 * pad_d, rows - 2 * pad_r, cols - 2 * pad_c), dtype=data.dtype)

    # KERNEL setup
    flat_kernel, kernel_mask = _kernel_setup(kernel)

    for i in prange(rows - 2 * pad_r):  # over rows (1024) instead of cols (128) or depth (36)
        for j in range(cols - 2 * pad_c):
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
                    results[d, i, j] = _weighted_median(valid_values, valid_weights)
    return results#type:ignore

def sliding_weighted_median_nd[Data: np.ndarray](data: Data, kernel: np.ndarray) -> Data:
    """
    To compute the sliding median for a weighted kernel and n-dimensional data.

    Args:
        data (np.ndarray): the n-dimensional data for which to get the sliding median.
        kernel (np.ndarray): the weighted kernel with the same dimensionality than the input data.

    Returns:
        np.ndarray: the sliding median results.
    """

    # SHAPE output
    output_shape = tuple(d - k + 1 for d, k in zip(data.shape, kernel.shape))

    # MEDIAN sliding
    result_flat = _sliding_weighted_median_nd(data, kernel)
    return result_flat.reshape(output_shape)#type:ignore

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

@njit
def _kernel_setup(
        kernel: np.ndarray[tuple[int, ...], np.dtype[np.floating]],
    ) -> tuple[
        np.ndarray[tuple[int], np.dtype[np.floating]],
        np.ndarray[tuple[int], np.dtype[np.bool_]],
    ]:
    """
    To flip the kernel (correlation vs convolution) and get the needed kernel mask.

    Args:
        kernel (np.ndarray[tuple[int, ...], np.dtype[np.floating]]): the kernel to setup.

    Returns:
        tuple[
            np.ndarray[tuple[int], np.dtype[np.floating]],
            np.ndarray[tuple[int], np.dtype[np.bool_]],
        ]: the flattened flipped kernel and the mask of valid weights.
    """

    # FLIP setup
    kernel_flip = kernel.copy()

    # FLIP kernel
    flat_kernel = kernel_flip.ravel()[::-1]
    valid_kernel_mask = (np.abs(flat_kernel) >= 1e-10)
    return flat_kernel, valid_kernel_mask#type:ignore

@njit
def _apply_kernel_weights(
        flat_window: np.ndarray,
        flat_kernel: np.ndarray[tuple[int], np.dtype[np.floating]],
        flat_kernel_not_nan: np.ndarray[tuple[int], np.dtype[np.bool_]],
    ) -> tuple[np.ndarray, np.ndarray]:
    """
    Gives the valid values and weights for a given window.

    Args:
        flat_window (np.ndarray): the flatten window to apply the kernel on.
        flat_kernel (np.ndarray[tuple[int], np.dtype[np.floating]]): the flattened kernel.
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

    if total_weight == 0.: return np.nan  # ? do I need to check if isfinite ?

    # STATISTICAL MEDIAN - 50% of total weight
    median_weight = total_weight / 2.

    # INDEX
    idx = np.searchsorted(cumsum, median_weight)
    if idx >= len(sorted_values): idx = len(sorted_values) - 1

    if idx + 1 < len(sorted_values) and np.isclose(cumsum[idx], median_weight):
        return 0.5 * (sorted_values[idx] + sorted_values[idx + 1])
    return sorted_values[idx]

@njit(parallel=True)
def _tuple_sliding_nanmedian_nd(
        data: np.ndarray,
        kernel: tuple[int, ...],
        output_shape: np.ndarray,
    ) -> np.ndarray:
    """
    To get the sliding median value for a n-dimensional input data and kernel.
    Padding must be done beforehand to handle borders correctly.

    Args:
        data (np.ndarray): the padded data to get the sliding median for. Can and should contain
            NaNs.
        kernel (tuple[int, ...]): the shape of the kernel.
        output_shape (np.ndarray): the shape of the output array.

    Returns:
        np.ndarray: the sliding median result.
    """

    ndim = len(kernel)
    data_shape = data.shape

    # SIZE output
    output_size = 1
    for s in output_shape: output_size *= s
    if output_size == 0: return np.empty(0, dtype=data.dtype)  # ? should I ?

    # STRIDEs
    data_strides = np.empty(ndim, dtype=np.int64)
    data_strides[-1] = 1
    for i in range(ndim - 2, -1, -1): data_strides[i] = data_strides[i + 1] * data_shape[i + 1]

    # SHAPE as array
    kernel_shape = np.empty(ndim, dtype=np.int64)
    for i in range(ndim): kernel_shape[i] = kernel[i]

    kernel_size = 1
    for s in kernel_shape: kernel_size *= s

    # OFFSETs for kernel
    kernel_offsets = np.empty(kernel_size, dtype=np.int64)
    for k in range(kernel_size):
        temp = k
        offset = 0
        for d in range(ndim - 1, -1, -1):
            coord = temp % kernel_shape[d]
            temp //= kernel_shape[d]
            offset += coord * data_strides[d]
        kernel_offsets[k] = offset
    results = np.empty(output_size, dtype=data.dtype)

    for i0 in prange(output_shape[0]):
        rest_size = 1
        for i in range(1, ndim): rest_size *= output_shape[i]

        for rest_idx in range(rest_size):
            # COORDINATEs output
            out_coords = np.empty(ndim, dtype=np.int64)
            out_coords[0] = i0
            temp = rest_idx
            for d in range(ndim - 1, 0, -1):
                out_coords[d] = temp % output_shape[d]
                temp //= output_shape[d]

            # INDEX result
            res_idx = 0
            res_strides = 1
            for d in range(ndim - 1, -1, -1):
                res_idx += out_coords[d] * res_strides
                if d > 0: res_strides *= output_shape[d]

            # ORIGIN index data
            origin_flat = 0
            for d in range(ndim): origin_flat += out_coords[d] * data_strides[d]

            # EXTRACT window
            window = np.empty(kernel_size, dtype=data.dtype)
            for k in range(kernel_size): window[k] = data.flat[origin_flat + kernel_offsets[k]]

            results[res_idx] = _fast_median(window)
    return results

@njit(parallel=True)
def _sliding_weighted_median_nd(data: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    To get the weighted sliding median for a n-dimensional input data and kernel.
    Padding must be done beforehand to handle borders correctly.

    Args:
        data (np.ndarray): the padded data to get the sliding median for. Can and should contain
            NaNs.
        kernel (np.ndarray): the weighted kernel.

    Returns:
        np.ndarray: the sliding median result.
    """

    ndim = data.ndim
    data_shape = data.shape
    kernel_shape = kernel.shape

    # SHAPE output
    output_shape = np.empty(ndim, dtype=np.int64)
    for i in range(ndim):
        output_shape[i] = data_shape[i] - kernel_shape[i] + 1

    # SIZE output
    output_size = 1
    for s in output_shape: output_size *= s
    if output_size == 0: return np.empty(0, dtype=data.dtype)  # ? should I ?
    results = np.empty(output_size, dtype=data.dtype)

    # STRIDEs result
    result_strides = np.empty(ndim, dtype=np.int64)
    result_strides[-1] = 1
    for i in range(ndim - 2, -1, -1):
        result_strides[i] = result_strides[i + 1] * output_shape[i + 1]

    # STRIDEs data
    data_strides = np.empty(ndim, dtype=np.int64)
    data_strides[-1] = 1
    for i in range(ndim - 2, -1, -1): data_strides[i] = data_strides[i + 1] * data_shape[i + 1]

    # KERNEL flip
    flat_kernel, kernel_mask = _kernel_setup(kernel)
    kernel_size = flat_kernel.size

    # OFFSETs for kernel
    kernel_offsets = np.empty(kernel_size, dtype=np.int64)
    for k in range(kernel_size):
        temp = k
        offset = 0
        for d in range(ndim - 1, -1, -1):
            coord = temp % kernel_shape[d]
            temp //= kernel_shape[d]
            offset += coord * data_strides[d]
        kernel_offsets[k] = offset

    for i0 in prange(output_shape[0]):
        rest_size = 1
        for i in range(1, ndim): rest_size *= output_shape[i]

        for rest_idx in range(rest_size):
            # COORDINATEs output
            out_coords = np.empty(ndim, dtype=np.int64)
            out_coords[0] = i0
            temp = rest_idx
            for d in range(ndim - 1, 0, -1):  # from last dim down to dim 1
                out_coords[d] = temp % output_shape[d]
                temp //= output_shape[d]

            # INDEX result
            res_idx = 0
            for d in range(ndim): res_idx += out_coords[d] * result_strides[d]

            # ORIGIN index data
            origin_flat = 0
            for d in range(ndim): origin_flat += out_coords[d] * data_strides[d]

            # EXTRACT window
            window_vals = np.empty(kernel_size, dtype=data.dtype)
            for k in range(kernel_size):
                window_vals[k] = data.flat[origin_flat + kernel_offsets[k]]

            # WEIGHTED MEDIAN
            valid_data = window_vals[kernel_mask]
            valid_weights = flat_kernel[kernel_mask]
            good = ~np.isnan(valid_data)
            if not np.any(good):
                results[res_idx] = np.nan
            else:
                values = valid_data[good]
                weights = valid_weights[good]
                sorted_idx = np.argsort(values)
                sorted_vals = values[sorted_idx]
                sorted_w = weights[sorted_idx]
                cumsum_w = np.cumsum(sorted_w)
                total_w = cumsum_w[-1]

                if total_w == 0.0 or not np.isfinite(total_w):
                    results[res_idx] = np.nan
                else:
                    threshold = total_w / 2.0
                    idx = np.searchsorted(cumsum_w, threshold)
                    if idx >= len(sorted_vals):
                        idx = len(sorted_vals) - 1

                    if idx + 1 < len(sorted_vals) and np.isclose(cumsum_w[idx], threshold):
                        results[res_idx] = 0.5 * (sorted_vals[idx] + sorted_vals[idx + 1])
                    else:
                        results[res_idx] = sorted_vals[idx]
    return results
