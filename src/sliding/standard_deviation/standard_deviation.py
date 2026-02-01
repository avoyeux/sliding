"""
Code to compute the sliding standard deviation given data (with/without NaNs) and kernel
(with/without weights).
"""
from __future__ import annotations

# IMPORTs alias
import numpy as np

# IMPORTs sub
from numpy.lib.stride_tricks import sliding_window_view

# IMPORTs local
from sliding.convolution import BorderType, Padding

# TYPE ANNOTATIONs
import numpy.typing as npt
from typing import Any

# API public
__all__ = ["SlidingStandardDeviation"]

# todo rewrite the _check_kernel method to make it a little cleaner



class SlidingStandardDeviation[Data: npt.NDArray[np.floating[Any]]]:
    """
    To compute the sliding standard deviations for data (with/without NaNs) using a kernel
    (with/without weights).
    The inputs need to be of np.floating type when using ndarrays (float64 recommended).
    The kernel must have odd dimensions.
    The sliding mean is also computed at the same time.
    """

    def __init__(
            self,
            data: Data,
            kernel: int | tuple[int, ...] | Data,
            borders: BorderType = 'reflect',
        ) -> None:
        """
        Computes the sliding standard deviations for data (with/without NaNs) using a kernel
        (with/without weights).
        The inputs need to be of np.floating type when using ndarrays (float64 recommended).
        The kernel must have odd dimensions.
        Given the way the standard deviation is computed (using a stable solution, c.f.
        '_get_standard_deviation'), the sliding mean is also computed at the same time as an
        intermediate step.
        To retrieve the computed standard deviations, use the 'standard_deviation' property.
        To retrieve the computed sliding mean, use the 'mean' property.

        Args:
            data (Data): the data for which the sliding standard deviations is computed.
                Needs to be a numpy ndarray of the same floating type than the kernel (if given as
                a numpy ndarray).
            kernel (int, tuple[int, ...] | Data): the kernel information. If an int, you have a
                'square' kernel. If a tuple, you are deciding on the shape of the kernel. If a
                numpy ndarray, you are giving the full kernel (can contain different weights). Keep
                in mind that the kernel should have the same dimensions and dtype as the data.
                Furthermore, all kernel dimensions must be positive odd integers.
            borders (BorderType, optional): the type of borders to use. These are the type of
                borders used by OpenCV (not all OpenCV borders are implemented as some don't have
                the equivalent in np.pad or scipy.ndimage). If None, uses adaptative borders, i.e.
                no padding and hence smaller kernels at the borders. Defaults to 'reflect'.
        """

        self._data = data
        self._kernel = self._check_kernel(kernel)
        self._borders = borders

        # RUN
        self._mean, self._standard_deviation = self._get_standard_deviation()

    @property
    def standard_deviation(self) -> Data:
        """
        Returns the sliding standard deviations.

        Returns:
            np.ndarray[tuple[int, ...], np.dtype[np.floating]]: the sliding standard deviation.
        """
        return self._standard_deviation

    @property
    def mean(self) -> Data:
        """
        Returns the sliding mean.

        Returns:
            np.ndarray[tuple[int, ...], np.dtype[np.floating]]: the sliding mean.
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

    def _get_standard_deviation(self) -> tuple[Data, Data]:
        """
        Computes the sliding standard deviation using a numerically stable solution.
        Given the operation done, the sliding mean is also computed at the same time.

        Operation done is basically:
            * n = len(data)
            * mean = sum(data) / n
            * variance = sum((x - mean) ** 2 for x in data) / (n - 1)
            * std = sqrt(variance)

        Of course, computations are done using arrays and sliding windows to do the 'for x in data'
        part as data is defined by a kernel / is a window.
        Furthermore, conditional operations are added to take care of NaN values in the data (or 
        from the padding).
        The memory usage is basically as low as possible when wanting to do all operations at once
        (still high because at least one full sliding window view buffer needs to be used).

        Returns:
            tuple[Data, Data]: the sliding mean and standard deviation.
        """

        # PAD data
        padded = Padding(
            data=self._data,
            kernel=self._kernel.shape,
            borders=self._borders,#type:ignore
        ).padded

        # NaN handling
        valid = ~np.isnan(padded)
        arr_filled = np.where(valid, padded, 0.0).astype(self._data.dtype)

        # VIEWs of the convolution space
        axis = tuple(range(self._data.ndim))
        windows = sliding_window_view(
            x=arr_filled,
            window_shape=self._kernel.shape,
            axis=axis,#type:ignore
        )
        valid_windows = sliding_window_view(
            x=valid,
            window_shape=self._kernel.shape,
            axis=axis,#type:ignore
        )

        # BUFFER window-shaped intermediates (huge memory usage)
        buffer = np.empty(windows.shape, dtype=self._data.dtype)

        kernel_axes = tuple(range(-len(axis), 0))
        uniform_kernel = np.allclose(self._kernel, self._kernel.flat[0])
        if uniform_kernel:
            # NO WEIGHTs
            # COUNT of valid data
            buffer[...] = valid_windows
            count = buffer.sum(axis=kernel_axes)

            # SUM sliding without NaNs
            np.copyto(buffer, windows)
            sum_sliding = buffer.sum(axis=kernel_axes)
        else:
            # WEIGHTED
            # COUNT valid data weighted
            np.multiply(valid_windows, self._kernel, out=buffer)
            count = buffer.sum(axis=kernel_axes)

            # WEIGHTED SUM sliding without NaNs
            np.multiply(windows, self._kernel, out=buffer)
            sum_sliding = buffer.sum(axis=kernel_axes)

        # MEAN sliding
        with np.errstate(divide="ignore", invalid="ignore"):
            mean = np.where(count > 0, sum_sliding / count, 0.0).astype(self._data.dtype)

        # VARIANCE windows
        mean_expanded = np.expand_dims(mean, axis=kernel_axes)
        np.subtract(windows, mean_expanded, out=buffer)

        # NaN handling
        buffer[~valid_windows] = 0.

        # DIFF**2 windows
        np.square(buffer, out=buffer)
        if not uniform_kernel: np.multiply(buffer, self._kernel, out=buffer)

        # SUM final
        M2 = buffer.sum(axis=kernel_axes)

        # STD 
        with np.errstate(divide="ignore", invalid="ignore"):
            std = np.where(count > 0, np.sqrt(M2 / count), 0.0).astype(self._data.dtype)
        return mean, std#type:ignore
