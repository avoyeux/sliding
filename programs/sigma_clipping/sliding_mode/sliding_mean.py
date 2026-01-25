"""
Code to compute the sliding mean given a kernel.
"""
from __future__ import annotations

# IMPORTs alias
import numpy as np
import numpy.typing as npt

# IMPORTs local
from programs.sigma_clipping.convolution import BorderType, Convolution

# TYPE ANNOTATIONs
from typing import cast, Any
type Array[D: np.floating] = npt.NDArray[D]

# API public
__all__ = ["SlidingMean"]



class SlidingMean[Data: Array[np.floating[Any]]]:
    """
    To compute the sliding mean of a given ndarray data and kernel.
    The inputs need to be of float32 or float64 type.
    """

    def __init__(
        self,
        data: Data,
        kernel: Data | tuple[int, ...],
        borders: BorderType = "reflect",
        threads: int | None = 1,
    ) -> None:
        """
        Computes the sliding mean of a given ndarray data and kernel.
        Weights can be used inside the kernel and the input data can have np.nan values.
        To access the sliding mean result, use the `mean` property.
        ! make sure that the kernel and data have the same dtype to avoid unwanted behaviors.

        Args:
            data (Data): the input data to compute the sliding mean from. Needs to be of float32
                or float64 type.
            kernel (Data, tuple[int, ...]): the kernel to use for the sliding mean computation.
                Can contain weights. If a numpy array, use the same dtype as the input data.
            borders (BorderType, optional): the type of borders to use. These are the type of
                borders used by OpenCV (not all OpenCV borders are implemented as some don't
                have the equivalent in np.pad or scipy.ndimage). If None, uses adaptative borders,
                i.e. no padding and hence smaller kernels at the borders. Defaults to 'reflect'.
            threads (int | None, optional): the number of threads to use for the computation.
                If None, doesn't change change the default behaviour. Defaults to 1.
        """

        self._data = data
        self._kernel = np.ones(kernel, dtype=data.dtype) if isinstance(kernel, tuple) else kernel
        self._borders = borders
        self._threads = threads

        # COMPUTE
        self._mean = self._sliding_mean()

    @property
    def mean(self) -> Data:
        """
        The sliding mean.
        Please do make sure that the kernel and input data have the same dtype  as to not create
        unwanted behaviors.

        Returns:
            Data: the sliding mean result.
        """
        return self._mean

    def _get_not_NaNs_count(
            self,
        ) -> tuple[np.ndarray[tuple[int, ...], np.dtype[np.bool_]], Array] | None:
        """
        To get the mask of non nan values and the count of nan values for each sliding window.
        Done so that the nan values can be swapped with 0. The result of the sliding mean is then
        corrected with this count. If there is no NaN in the data, returns None.

        Returns:
            tuple[np.ndarray[tuple[int, ...], np.dtype[np.bool_]], Array] | None: the mask of valid
                data and the sliding window count of nan values.
        """

        # TYPE CHECKER complains
        self._borders = cast(BorderType, self._borders)

        # NaN handling
        if (isnan := np.isnan(self._data)).any():
            valid_mask = ~isnan

            # COUNT NaN
            count = Convolution(
                data=valid_mask.astype(self._data.dtype),
                kernel=np.ones(self._kernel.shape, dtype=self._data.dtype),
                borders=self._borders,
                threads=self._threads,
            ).result
            return valid_mask, count
        return None

    def _sliding_mean(self) -> Data:
        """
        Computes the sliding mean.

        Returns:
            Data: the sliding mean values as an array of np.ndarray. The length of the list is 1
                if the input data was a single array.
        """

        # TYPE CHECKER complains
        self._borders = cast(BorderType, self._borders)

        with_NaNs = self._get_not_NaNs_count()
        if with_NaNs is None:
            kernel_norm = self._kernel / self._kernel.sum()

            # MEAN
            means = Convolution(
                data=self._data,
                kernel=kernel_norm,
                borders=self._borders,
                threads=self._threads,
            ).result
            return means

        # NaN handling
        valid_mask, count = with_NaNs

        # NaN handling
        data_filled = np.where(valid_mask, self._data, 0.).astype(self._data.dtype)

        # SUM
        sum_values = Convolution(
            data=data_filled,
            kernel=self._kernel,
            borders=self._borders,
            threads=self._threads,
        ).result
        with np.errstate(divide='ignore', invalid='ignore'):
            means = np.where(count > 0, sum_values / count, 0.0).astype(self._data.dtype)
        return cast(Data, means)

if __name__ == "__main__":
    data = np.ones((10, 10), dtype=np.float32)
    instance = SlidingMean(
        data=data,
        kernel=np.ones((3, 3), dtype=np.float32),
        borders='reflect',
        threads=1,
    )
    mean = instance.mean
