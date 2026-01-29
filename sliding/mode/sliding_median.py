"""
Code to compute the sliding median given an ndarray data and a tuple or ndarray kernel.
"""
from __future__ import annotations

# IMPORTs alias
import numpy as np

from numba import set_num_threads

# IMPORTs local
from sliding.convolution import Padding, BorderType
from sliding.mode.numba_functions import (
    tuple_sliding_nanmedian_3d, tuple_sliding_nanmedian_nd,
    sliding_weighted_median_3d, sliding_weighted_median_nd,
)

# TYPE ANNOTATIONs
import numpy.typing as npt
from typing import Any

# API public
__all__ = ["SlidingMedian"]

# ? should I add a dimension check to ensure that the kernel and data have compatible dimensions?



class SlidingMedian[Data: npt.NDArray[np.floating[Any]]]:
    """
    To compute the sliding median of a given ndarray data and kernel.
    The inputs need to be of np.floating type.
    """

    def __init__(
            self,
            data: Data,
            kernel: Data | tuple[int, ...],
            borders: BorderType = "reflect",
            threads: int | None = 1,
        ) -> None:
        """
        Computes the sliding median of a given ndarray data and a tuple or ndarray kernel.
        The inputs need to be of np.floating type when using ndarrays.
        To access the sliding median result, use the `median` property.

        Args:
            data (Data): the input data to compute the sliding median from.
            kernel (Data | tuple[int, ...]): the kernel to use for the sliding median computation.
                When given as an ndarray, it is supposed to contain weights. If no weights given,
                use a tuple of integers representing the kernel shape.
            borders (BorderType, optional): the type of borders to use. These are the type of
                borders used by OpenCV (not all OpenCV borders are implemented as some don't
                have the equivalent in np.pad or scipy.ndimage). If None, uses adaptative borders,
                i.e. no padding and hence smaller kernels at the borders. Defaults to 'reflect'.
            threads (int | None, optional): the number of threads to use for the computation.
                If None, doesn't change change the default behaviour. Defaults to 1.
        """

        self._data = data
        self._kernel = kernel
        self._borders = borders
        self._threads = threads

        # RUN
        if self._threads is not None: set_num_threads(self._threads)
        self._sliding_median = self._get_sliding_median()

    @property
    def median(self) -> Data:
        """
        To access the sliding median result.

        Returns:
            np.ndarray: the sliding median result.
        """
        return self._sliding_median

    def _get_sliding_median(self) -> Data:
        """
        Depending on the kernel type and dimensions, chooses the correct numba function, adds
        padding to the data and computes the sliding median.

        Returns:
            np.ndarray: the sliding median results.
        """

        # PADDING
        padded = Padding(
            data=self._data,
            kernel=self._kernel.shape if isinstance(self._kernel, np.ndarray) else self._kernel,
            borders=self._borders,#type:ignore
        ).padded

        # SLIDING MEDIAN
        if isinstance(self._kernel, tuple):
            if len(self._kernel) == 3: return tuple_sliding_nanmedian_3d(padded, self._kernel)
            return tuple_sliding_nanmedian_nd(padded, self._kernel)
        else:
            if len(self._kernel.shape) == 3:
                return sliding_weighted_median_3d(padded, self._kernel)
            return sliding_weighted_median_nd(padded, self._kernel)



if __name__ == "__main__":
    data = np.ones((10, 10), dtype=np.float64)
    instance = SlidingMedian(
        data=data,
        kernel=(3, 3),
        borders='reflect',
        threads=1,
    )
    median = instance.median
