"""
Code to compute the sliding mean given a kernel.
"""
from __future__ import annotations

# IMPORTs alias
import numpy as np
import numpy.typing as npt

# IMPORTs local
from programs.standard_deviation import BorderType, Convolution  # ! need to change the path to this code

# TYPE ANNOTATIONs
from typing import cast
type Array[D: np.floating] = npt.NDArray[D]
type ArrayLike[D: np.floating] = Array[D] | list[Array[D]]

# API public
__all__ = ["SlidingMean"]



class SlidingMean[Data: ArrayLike]:
    """
    To compute the sliding mean of a given ndarray data and kernel.
    The inputs need to be of float32 or float64 type.
    """

    def __init__(
        self,
        data: Data,
        kernel: Array,
        borders: BorderType = "reflect",
        threads: int = 1,
    ) -> None:
        """
        Computes the sliding mean of a given ndarray data and kernel.
        Weights can be used inside the kernel and the input data can have np.nan values.
        To access the sliding mean result, use the `sliding_mean` property.
        When the input data is given as a list of arrays, is it supposed that the positions of
        np.nan are the same for all arrays. Furthermore, the kernel used remains the same for all
        arrays.
        # ! make sure that the kernel and data have the same dtype to avoid unwanted behaviors.

        Args:
            data (Data): the input data (or list of data) to compute the sliding mean from.
            kernel (Array): the kernel to use for the sliding mean computation. Can contain
                weights.
            borders (BorderType, optional): the type of borders to use. These are the type of
                borders used by OpenCV (not all OpenCV borders are implemented as some don't
                have the equivalent in np.pad or scipy.ndimage). If None, uses adaptative borders,
                i.e. no padding and hence smaller kernels at the borders. Defaults to 'reflect'.
            threads (int | None, optional): the number of threads to use for the computation.
                If None, doesn't change change the default behaviour. Defaults to 1.
        """

        self._data = cast(list[Array], data if isinstance(data, list) else [data])
        self._kernel = kernel
        self._borders = borders
        self._threads = threads

        # COMPUTE
        self._mean = self._sliding_mean()

    @property
    def sliding_mean(self) -> Data:
        """
        Gives the sliding mean results.
        Keep in mind that, if the input data was a list of arrays, then the results is also a list
        of array with the same type and shape than the input data. Please do make sure that the
        kernel and input data have the same dtype to not create unwanted behaviors.

        Returns:
            Data: _description_
        """

        if len(self._mean) == 1: return cast(Data, self._mean[0])
        return self._mean

    def _get_not_NaNs_count(self) -> tuple[np.ndarray[tuple[int, ...], np.dtype[np.bool_]], Array]:
        """
        To get the mask of non nan values and the count of nan values for each sliding window.
        Done so that the nan values can be swapped with 0. The result of the sliding mean is then
        corrected with this count.

        Returns:
            tuple[np.ndarray[tuple[int, ...], np.dtype[np.bool_]], Array]: the mask of valid data
                and the sliding window count of nan values.
        """

        # ! if I normalise the kernel, no need for this if there is no NaN

        # TYPE CHECKER complains
        self._borders = cast(BorderType, self._borders)

        # NaN handling
        valid_mask = ~np.isnan(self._data[0])  # even if no NaNs need it for border effects

        # COUNT NaN
        count = Convolution(
            data=valid_mask.astype(self._data[0].dtype),
            kernel=np.ones(self._kernel.shape, dtype=self._data[0].dtype),
            borders=self._borders,
            threads=self._threads,
        ).result
        return valid_mask, count

    def _sliding_mean(self) -> Data:
        """
        Computes the sliding mean.

        Returns:
            Data: the sliding mean values as an array of np.ndarray. The length of the list is 1
                if the input data was a single array.
        """

        # TYPE CHECKER complains
        self._borders = cast(BorderType, self._borders)

        # NaN handling
        nan_mask, count = self._get_not_NaNs_count()

        # MEMORY pre-allocation
        means: Data = cast(Data, [None] * len(self._data))

        for i, data in enumerate(self._data):

            # NaN handling
            if nan_mask is not None:
                data_filled = np.where(nan_mask, data, 0.).astype(data.dtype)
            else:
                data_filled = data

            # SUM
            sum_values = Convolution(
                data=data_filled,
                kernel=self._kernel,
                borders=self._borders,
                threads=self._threads,
            ).result
            with np.errstate(divide='ignore', invalid='ignore'):
                means[i] = np.where(count > 0, sum_values / count, 0.0).astype(data.dtype)
        return means



if __name__ == "__main__":
    data = np.ones((10, 10), dtype=np.float32)
    instance = SlidingMean(
        data=data,
        kernel=np.ones((3, 3), dtype=np.float32),
        borders='reflect',
        threads=1,
    )
    mean = instance.sliding_mean
