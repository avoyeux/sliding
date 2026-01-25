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
from typing import cast

# API public
__all__ = ["FastStandardDeviation"]



class FastStandardDeviation[Data: np.ndarray[tuple[int, ...], np.dtype[np.floating]]]:
    """
    To compute the moving sample standard deviations using convolutions.
    """

    def __init__(
            self,
            data: Data,
            kernel: int | tuple[int, ...] | Data,
            borders: BorderType = 'reflect',
            threads: int | None = 1,
        ) -> None:
        """
        Computes the moving sample standard deviations using convolutions. The size of each sample
        is defined by the kernel. If you decide to have different weights in the kernel, keep in
        mind that the standard deviation will take a little longer to compute (no approximation
        possible).
        To retrieve the computed standard deviations, use the 'sdev' property.
        NaN handling is done.
        ! IMPORTANT: the kernel size must be odd and of the same dimensionality as the input array
        ! (when the kernel is given as an ndarray or a tuple of ints).

        Args:
            data (Data): the data for which the moving sample standard deviations are computed.
                Needs to be a numpy ndarray of the same floating type than the kernel (if given as
                a numpy ndarray).
            kernel (int, tuple[int, ...] | Data): the kernel information. If an int, you have a
                'square' kernel. If a tuple, you are deciding on the shape of the kernel. If a
                numpy ndarray, you are giving the full kernel (can contain different weights). Keep
                in mind that the kernel should have the same dimensions and dtype as the data.
            borders (BorderType, optional): the type of borders to use. These are the type of
                borders used by OpenCV (not all OpenCV borders are implemented as some don't have
                the equivalent in np.pad or scipy.ndimage). If None, uses adaptative borders, i.e.
                no padding and hence smaller kernels at the borders. Defaults to 'reflect'.
            threads (int | None, optional): the number of threads to use for the computation.
                If None, doesn't change any thread values. Defaults to 1.
                ! Might not work as expected given that numpy, numba and cv2 do not always let you
                ! set the number of threads at runtime.
        """

        self._data = data
        self._borders = borders
        self._threads = threads

        # CHECK kernel
        self._kernel = self._check_kernel(kernel)

        # RUN
        self._sdev = self._sdev_loc()

    @property
    def sdev(self) -> Data:
        """
        Returns the moving sample standard deviations.

        Returns:
            np.ndarray[tuple[int, ...], np.dtype[np.floating]]: Array of moving sample standard
                deviations.
        """
        return self._sdev

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

    def _sdev_loc(self) -> Data:
        """
        Computes the sliding standard deviation using the Welford Formula.

        Returns:
            Data: the sliding standard deviation.
        """

        axis = tuple(range(self._data.ndim))

        # PAD data
        padded = Padding(
            data=self._data,
            kernel=self._kernel.shape,
            borders=self._borders,#type:ignore
        ).padded

        # NaN handling
        valid = ~np.isnan(padded)
        arr_filled = np.where(valid, padded, 0.).astype(self._data.dtype)

        # SLIDING WINDOWS
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

        # COUNT weighted (sum of weights for valid entries)
        kernel_axes = tuple(range(-len(axis), 0))
        kernel_sum = (valid_windows * self._kernel).sum(axis=kernel_axes)

        # MEAN weighted
        sum_vals = (windows * self._kernel).sum(axis=kernel_axes)
        with np.errstate(divide="ignore", invalid="ignore"):
            mean = np.where(kernel_sum > 0, sum_vals / kernel_sum, 0.0).astype(self._data.dtype)

        # STD weighted
        diff = np.where(valid_windows, windows - np.expand_dims(mean, axis=kernel_axes), 0.0)
        M2 = (diff**2 * self._kernel).sum(axis=kernel_axes)
        with np.errstate(divide="ignore", invalid="ignore"):
            std = np.where(kernel_sum > 0, np.sqrt(M2 / kernel_sum), 0.0).astype(self._data.dtype)
        return cast(Data, std)

    # def _sdev_loc_old(self) -> Data:  # ! problems with low values
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
