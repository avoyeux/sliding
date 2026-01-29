"""
Contains the code that needs testing to see if it can be used for SPICE (for moving sample
standard deviation calculations).
"""
from __future__ import annotations

# IMPORTs alias
import numpy as np

# IMPORTs sub
from numpy.lib.stride_tricks import sliding_window_view

# IMPORTs local
from programs.sigma_clipping.convolution import BorderType, Padding

# TYPE ANNOTATIONs
import numpy.typing as npt
from typing import Any

# API public
__all__ = ["SlidingStandardDeviation"]



class SlidingStandardDeviation[Array: npt.NDArray[np.floating[Any]]]:
    """
    To compute the moving sample standard deviations using convolutions.
    """

    def __init__(
            self,
            data: Array,
            kernel: int | tuple[int, ...] | Array,
            borders: BorderType = 'reflect',
        ) -> None:
        """
        Computes the moving sample standard deviations using convolutions.
        Given the way the standard deviation is computed (using a stable solution, c.f.
        '_standard_deviation'), the sliding mean is also computed at the same time as an
        intermediate step.
        The size of each sample is defined by the kernel.
        To retrieve the computed standard deviations, use the 'standard_deviation' property.
        To retrieve the computed sliding mean, use the 'mean' property.
        NaN handling is done.
        ! IMPORTANT: the kernel size must be odd and of the same dimensionality as the input array.
        ! IMPORTANT: recommended to use float64 even if float32 is supported. 

        Args:
            data (Array): the data for which the moving sample standard deviations are computed.
                Needs to be a numpy ndarray of the same floating type than the kernel (if given as
                a numpy ndarray).
            kernel (int, tuple[int, ...] | Array): the kernel information. If an int, you have a
                'square' kernel. If a tuple, you are deciding on the shape of the kernel. If a
                numpy ndarray, you are giving the full kernel (can contain different weights). Keep
                in mind that the kernel should have the same dimensions and dtype as the data.
            borders (BorderType, optional): the type of borders to use. These are the type of
                borders used by OpenCV (not all OpenCV borders are implemented as some don't have
                the equivalent in np.pad or scipy.ndimage). If None, uses adaptative borders, i.e.
                no padding and hence smaller kernels at the borders. Defaults to 'reflect'.
        """

        self._data = data
        self._borders = borders

        # CHECK kernel
        self._kernel = self._check_kernel(kernel)

        # RUN
        self._mean, self._sdev = self._standard_deviation()

    @property
    def standard_deviation(self) -> Array:
        """
        Returns the moving sample standard deviations.

        Returns:
            np.ndarray[tuple[int, ...], np.dtype[np.floating]]: Array of moving sample standard
                deviations.
        """
        return self._sdev

    @property
    def mean(self) -> Array:
        """
        Returns the moving sample means.

        Returns:
            np.ndarray[tuple[int, ...], np.dtype[np.floating]]: Array of moving sample means.
        """
        return self._mean

    def _check_kernel(self, kernel: int | tuple[int, ...] | np.ndarray) -> np.ndarray:
        """
        To check and create the kernel as a numpy array if needed.

        Args:
            kernel (int | tuple[int, ...] | np.ndarray): the kernel.

        Raises:
            ValueError: if the kernel does not match data dimensions.
            TypeError: if the kernel type is not supported.

        Returns:
            np.ndarray: the kernel as a numpy array.
        """

        if isinstance(kernel, np.ndarray):
            if kernel.ndim != self._data.ndim:
                raise ValueError(
                    "If 'kernel' is given as a numpy ndarray, it must have the same number of "
                    "dimensions as 'data'."
                )
            if np.isnan(kernel).any():
                raise ValueError("The kernel numpy ndarray must not contain any NaN values.")
            return kernel / kernel.sum()
        elif isinstance(kernel, tuple):
            if len(kernel) != self._data.ndim:
                raise ValueError(
                    "If 'kernel' is given as a tuple, it must have the same number of elements "
                    "as 'data' has dimensions."
                )
            if not all(isinstance(k, int) for k in kernel):
                raise TypeError("All elements of 'kernel' tuple must be of type int.")
            return np.ones(kernel, dtype=self._data.dtype) / np.prod(kernel)
        elif isinstance(kernel, int):
            normalised = (
                np.ones((kernel,) * self._data.ndim, dtype=self._data.dtype) /
                (kernel ** self._data.ndim)
            )
            return normalised
        else:
            raise TypeError(
                "'kernel' must be an int, a tuple of ints, or a numpy ndarray."
            )

    def _standard_deviation(self) -> tuple[Array, Array]:
        """
        Computes the sliding standard deviation using the same equation than in Welford.
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
            tuple[Array, Array]: the sliding mean and standard deviation.
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

    # def _sdev_loc_old(self) -> Array:  # ! problems with low values
    #     """
    #     Computes the moving sample standard deviations. The size of each sample is defined by the
    #     kernel (square with a length of 'size').

    #     Returns:
    #         np.ndarray[tuple[int, ...], np.dtype[np.floating]]: Array of moving sample standard
    #             deviations.
    #     """

    #     if (nan_mask := np.isnan(self._data)).any():
    #         # INSTABILITY helper
    #         global_median = np.nanmedian(self._data)
    #         shifted_data = self._data - global_median

    #         # VALID (non-NaN)
    #         valid_mask = ~nan_mask
    #         data_filled = np.where(valid_mask, shifted_data, 0.).astype(self._data.dtype)

    #         # SUM n MEAN
    #         sum_values = Convolution(
    #             data=data_filled,
    #             kernel=self._kernel,
    #             borders=self._borders,#type:ignore
    #             threads=self._threads
    #         ).result
    #         sum_squares = Convolution(
    #             data=data_filled ** 2,#type:ignore
    #             kernel=self._kernel,
    #             borders=self._borders,#type:ignore
    #             threads=self._threads,
    #         ).result
    #         count = Convolution(
    #             data=valid_mask.astype(self._data.dtype),
    #             kernel=self._kernel,
    #             borders=self._borders,#type:ignore
    #             threads=self._threads,
    #         ).result
    #         with np.errstate(divide='ignore', invalid='ignore'):
    #             mean = np.where(count > 0, sum_values / count, 0.0)
    #             mean_sq = np.where(count > 0, sum_squares / count, 0.0)

    #         # STD
    #         variance = mean_sq - mean ** 2
    #         variance = np.maximum(variance, 0.0)
    #         return np.sqrt(variance)#type:ignore
    #     else:
    #         # INSTABILITY helper
    #         global_median = np.median(self._data)
    #         shifted_data = self._data - global_median

    #         # STD
    #         mean2 = Convolution(
    #             data=shifted_data,
    #             kernel=self._kernel,
    #             borders=self._borders,#type:ignore
    #             threads=self._threads
    #         ).result ** 2
    #         variance = Convolution(
    #             data=shifted_data ** 2,
    #             kernel=self._kernel,
    #             borders=self._borders,#type:ignore
    #             threads=self._threads,
    #         ).result
    #         variance -= mean2
    #         variance = np.maximum(variance, 0.0)
    #         return np.sqrt(variance)#type:ignore
