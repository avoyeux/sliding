"""
Code to compute the sliding median given an ndarray data and a kernel.
"""
from __future__ import annotations

# IMPORTs alias
import numpy as np

from numba import set_num_threads

# IMPORTs local
from sliding.convolution import Padding, BorderType
from sliding.mode.numba_functions import sliding_weighted_median_nd

# API public
__all__ = ["SlidingMedian"]



class SlidingMedian:
    """
    To compute the sliding median of a given ndarray data and a kernel.
    When numpy arrays, the inputs need to be of np.floating type.
    The kernel must have odd dimensions.
    NaN handling is done internally.
    To access the sliding median result, use the `median` property.
    """

    def __init__(
            self,
            data: np.ndarray[tuple[int, ...], np.dtype[np.floating]],
            kernel: int | tuple[int, ...] | np.ndarray[tuple[int, ...], np.dtype[np.floating]],
            borders: BorderType = "reflect",
            threads: int | None = 1,
        ) -> None:
        """
        Computes the sliding median of a given ndarray data and a kernel.
        The inputs need to be of np.floating type when using ndarrays.
        The kernel must have odd dimensions.
        NaN handling is done internally.
        To access the sliding median result, use the `median` property.
        ! Do use the same dtype for both the kernel and the data to avoid unwanted behaviors.

        Args:
            data (np.ndarray[tuple[int, ...], np.dtype[np.floating]]): the input data to compute
                the sliding median from.
            kernel (int | tuple[int, ...] | np.ndarray[tuple[int, ...], np.dtype[np.floating]]):
                the kernel to use for the sliding median computation. When given as an ndarray, it
                can to contain weights. All kernel dimensions must be positive odd integers.
            borders (BorderType, optional): the type of borders to use. These are the type of
                borders used by OpenCV (not all OpenCV borders are implemented as some don't
                have the equivalent in np.pad or scipy.ndimage). If None, uses adaptative borders,
                i.e. no padding and hence smaller kernels at the borders. Defaults to 'reflect'.
            threads (int | None, optional): the number of threads to used by numba for the
                computation. If None, doesn't change change the default behaviour. Defaults to 1.
        """

        self._data = data
        self._kernel = self._check_kernel(kernel)
        self._borders = borders
        self._threads = threads

        # RUN
        if self._threads is not None: set_num_threads(self._threads)
        self._sliding_median = self._get_sliding_median()

    @property
    def median(self) -> np.ndarray[tuple[int, ...], np.dtype[np.floating]]:
        """
        To access the sliding median result.

        Returns:
            np.ndarray: the sliding median result.
        """
        return self._sliding_median

    def _check_kernel(self, kernel: int | tuple[int, ...] | np.ndarray) -> np.ndarray:
        """
        To check the input kernel shape, type and convert it to an ndarray if needed.

        Args:
            kernel (int | tuple[int, ...] | np.ndarray): the kernel to check.

        Raises:
            TypeError: if the kernel is not an int, a tuple of ints or an ndarray.
            ValueError: if the kernel shape is not composed of positive odd integers or if the
                kernel dimensions do not match the data dimensions.

        Returns:
            np.ndarrays: the kernel as an ndarray.
        """

        if isinstance(kernel, int):
            if kernel <=0 or kernel % 2 == 0:
                raise ValueError("The kernel size must be a positive odd integer.")
            return np.ones((kernel,) * self._data.ndim, dtype=self._data.dtype)#type:ignore
        elif isinstance(kernel, tuple):
            if any(k <=0 or k % 2 == 0 for k in kernel):
                raise ValueError("All kernel dimensions must be positive odd integers.")
            elif len(kernel) != self._data.ndim:
                raise ValueError(
                    "If 'kernel' is given as a tuple, it must have the same number of "
                    "dimensions as 'data'."
                )
            return np.ones(kernel, dtype=self._data.dtype)#type:ignore
        elif isinstance(kernel, np.ndarray):
            if any(s <=0 or s % 2 == 0 for s in kernel.shape):
                raise ValueError("All kernel dimensions must be positive odd integers.")
            elif kernel.ndim != self._data.ndim:
                raise ValueError(
                    "If 'kernel' is given as a numpy ndarray, it must have the same number of "
                    "dimensions as 'data'."
                )
            return kernel
        else:
            raise TypeError("The kernel must be an integer, a tuple of integers or an ndarray.")

    def _get_sliding_median(self) -> np.ndarray:
        """
        Adds the corresponding padding and computes the sliding median.

        Returns:
            np.ndarray: the sliding median results.
        """

        # PADDING
        padded = Padding(
            data=self._data,
            kernel=self._kernel.shape,
            borders=self._borders,#type:ignore
        ).padded

        # SLIDING MEDIAN
        return sliding_weighted_median_nd(padded, self._kernel)
