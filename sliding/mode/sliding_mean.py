"""
Code to compute the sliding mean given a kernel.
"""
from __future__ import annotations

# IMPORTs alias
import numpy as np
import numpy.typing as npt

# IMPORTs local
from sliding.convolution import BorderType, Convolution

# TYPE ANNOTATIONs
from typing import cast, Any

# API public
__all__ = ["SlidingMean"]



class SlidingMean[Data: npt.NDArray[np.floating[Any]]]:
    """
    To compute the sliding mean of a given ndarray data and kernel.
    The inputs should be of float32 or float64 type.
    Kernel dimensions must be positive odd integers.
    NaN handling is done internally.
    To access the sliding mean result, use the `mean` property.
    """

    def __init__(
        self,
        data: Data,
        kernel: int | tuple[int, ...] | Data,
        borders: BorderType = "reflect",
        threads: int | None = 1,
    ) -> None:
        """
        Computes the sliding mean of a given ndarray data and kernel.
        Weights can be used inside the kernel and the input data can have NaN values.
        To access the sliding mean result, use the `mean` property.
        NaN handling is done internally.
        ! make sure that the kernel and data have the same dtype to avoid unwanted behaviors.

        Args:
            data (Data): the input data to compute the sliding mean from. Needs to be of float32
                or float64 type.
            kernel (int | tuple[int, ...] | Data): the kernel to use for the sliding mean
                computation. Can contain weights. If a numpy array, use the same dtype as the input
                data. The kernel dimensions must be positive odd integers.
            borders (BorderType, optional): the type of borders to use. These are the type of
                borders used by OpenCV (not all OpenCV borders are implemented as some don't
                have the equivalent in np.pad or scipy.ndimage). If None, uses adaptative borders,
                i.e. padding with no values and hence smaller kernels at the borders.
                Defaults to 'reflect'.
            threads (int | None, optional): the number of threads to use for the computation.
                If None, doesn't change change the default behaviour. Defaults to 1.
        """

        self._data = data
        self._kernel = self._check_kernel(kernel)
        self._borders = borders
        self._threads = threads

        # COMPUTE
        self._mean = self._sliding_mean()

    @property
    def mean(self) -> Data:
        """
        The sliding mean.
        Please do make sure that the kernel and input data have the same dtype as to not create
        unwanted behaviors.

        Returns:
            Data: the sliding mean result.
        """
        return self._mean

    def _check_kernel(self, kernel: int | tuple[int, ...] | Data) -> Data:
        """
        To check the input kernel shape, type and convert it to an ndarray if needed.

        Args:
            kernel (int | tuple[int, ...] | Data): the kernel to check.

        Raises:
            TypeError: if the kernel is not an int, a tuple of ints or an ndarray.
            ValueError: if the kernel shape is not composed of positive odd integers or if the
                kernel dimensions do not match the data dimensions.

        Returns:
            Data: the kernel as an ndarray.
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

    def _get_not_NaNs_count(
            self,
        ) -> tuple[np.ndarray[tuple[int, ...], np.dtype[np.bool_]], Data] | None:
        """
        To get the mask of non nan values and the weighted sum of non nan values for each sliding
        window.
        Done so that the nan values can be swapped with 0. The result of the sliding mean is then
        corrected with this weighted sum. If there is no NaN in the data, returns None.

        Returns:
            tuple[np.ndarray[tuple[int, ...], np.dtype[np.bool_]], Data] | None: the mask of valid
                data and the sliding window weighted sum of non-NaN values. If there is no NaN
                in the data, returns None.
        """

        # TYPE CHECKER complains
        self._borders = cast(BorderType, self._borders)

        # NaN handling
        if (isnan := np.isnan(self._data)).any():
            valid_mask = ~isnan

            # EFFECTIVE WEIGHT SUM (sum of kernel weights over non-NaN entries)
            weight_sum = Convolution(
                data=valid_mask.astype(self._data.dtype),
                kernel=self._kernel.astype(self._data.dtype),
                borders=self._borders,
                cval=1. if self._borders == 'constant' else 0.,
                threads=self._threads,
            ).result
            return valid_mask, weight_sum#type:ignore
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
        valid_mask, weighted_valid_sum = with_NaNs

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
            means = np.where(
                weighted_valid_sum > 0,
                sum_values / weighted_valid_sum,
                0.0,
            ).astype(self._data.dtype)
        return cast(Data, means)
