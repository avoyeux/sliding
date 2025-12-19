"""
Contains the full sigma clipping implementation used in sospice.
"""
from __future__ import annotations

# IMPORTs alias
import numpy as np

# IMPORTs sub
from numpy import ma
from scipy.ndimage import generic_filter

# TYPE ANNOTATIONs
from typing import cast, Literal, Callable

import time


def _get_numpy_function(data: np.ndarray, name: str) -> Callable[..., np.ndarray | np.floating]:
    """
    Get NaN-aware version of numpy array function if there are any NaNs in data

    Parameters
    ----------
    data: numpy.array
        Data
    name: str
        Numpy function name, in its not NaN-aware version

    Return
    ------
    function
        Numpy function
    """

    if np.isnan(data).any(): name = "nan" + name
    return getattr(np, name)

def sigma_clip[T: np.ndarray](
        data: T,
        size: int | tuple[int, ...],
        sigma: float = 3,
        sigma_lower: float | None = None,
        sigma_upper: float | None = None,
        max_iters: int | None = 5,
        center_func: Literal['median', 'mean'] = 'median',
        masked: bool = True,
    ) -> T | ma.MaskedArray:
    """
    Performs sigma-clipping of the input array.

    Parameters
     ----------
    data: numpy.ndarray
        Input array
    size: int or tuple[int]
        Size of the kernel used to compute the running median (or mean) and standard deviation
    sigma: float
        The number of standard deviations to use for both the lower and upper clipping limit.
        This is overridden by `sigma_lower` and `sigma_upper`
    sigma_lower: float
        Low threshold, in units of the standard deviation of the local intensity distribution
    sigma_upper: float
        High threshold, in units of the standard deviation of the local intensity distribution
    max_iters: int
        Maximum number of iterations to perform. If None, iterate until convergence.
    center_func: str
        Method used to estimate the center of the local intensity distribution ("median" (default) or "mean")
    masked: bool
        Return a `numpy.ma.MaskedArray` (default) instead of an `numpy.array`

    Returns
    -------
    numpy.ndarray
        Filtered array, with clipped pixels replaced by the estimated value of the center of the
        local intensity distribution (either median or mean).
    """

    output = data.copy()
    if isinstance(size, int): size = (size,) * data.ndim

    sigma_lower = sigma_lower or sigma
    sigma_upper = sigma_upper or sigma
    max_iters = cast(int, max_iters or np.inf)
    n_changed = 1
    iteration = 0
    # PLACEHOLDER
    center = np.empty(1)
    while n_changed != 0 and (iteration < max_iters):
        iteration += 1
        start = time.time()
        center = generic_filter(output, _get_numpy_function(output, center_func), size)
        end = time.time()
        stddev = generic_filter(output, _get_numpy_function(output, "std"), size)
        end2 = time.time()
        print(f"{iteration} Center: {end - start:.2f} s")
        print(f"{iteration} Stddev: {end2 - end:.2f} s")
        diff = output - center
        new_mask = (diff > sigma_upper * stddev) | (diff < -sigma_lower * stddev)
        output[new_mask] = np.nan
        n_changed = np.count_nonzero(new_mask)  # ! .any() would be faster as short-circuiting

    nan = np.isnan(output)
    output[nan] = center[nan]
    if masked: return ma.masked_array(output, mask=nan)
    return output
