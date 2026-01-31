"""
Contains numba-optimized functions to compute the sliding median of n-dimensional arrays given
a kernel, with weights.
"""
from __future__ import annotations

# IMPORTs alias
import numpy as np

# IMPORTs sub
from numba import njit, prange

# API public
__all__ = ['sliding_weighted_median_nd']



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
