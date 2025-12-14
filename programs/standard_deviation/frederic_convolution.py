"""
Contains the code that needs testing to see if it can be used for SPICE (for moving sample
standard deviation calculations).
"""
from __future__ import annotations

# IMPORTs standard
import cv2

# IMPORTs alias
import numpy as np

# IMPORTs sub
from scipy.ndimage import convolve

# IMPORTs personal
from common import Decorators

# TYPE ANNOTATIONs
from typing import Any

# API public
__all__ = ["QuickSTDs"]



class QuickSTDs:
    """
    To compute the moving sample standard deviations using convolutions.
    """

    # @Decorators.running_time
    def __init__(
            self,
            data: np.ndarray[tuple[int, ...], np.dtype[Any]],
            kernel: int | tuple[int, ...] | np.ndarray[tuple[int, ...], np.dtype[np.float64]],
            with_NaNs: bool = True,
            cv2_threads: int | None = 1,
        ) -> None:
        """
        Computes the moving sample standard deviations using convolutions. The size of each sample
        is defined by the kernel. If you decide to choose to have different weights in the kernel,
        keep in mind that the standard deviation will take a little longer to compute (no
        approximation possible).
        To retrieve the computed standard deviations, use the 'sdev' property.

        Args:
            data (np.ndarray[tuple[int, ...], np.dtype[Any]]): the data for which the moving sample
                standard deviations are computed.
            kernel (int, tuple[int, ...] | np.ndarray[tuple[int, ...], np.dtype[np.float64]]): the
                kernel information. If an int, you have a 'square' kernel. If a tuple, you are
                deciding on the shape of the kernel. If a numpy ndarray, you are giving the full
                kernel (can contain different weights). Keep in mind that the kernel should have
                the same dimensions as the data.
            with_NaNs (bool, optional): whether to handle NaNs in the data. More efficient to set
                it to False. Defaults to True.
            cv2_threads (int | None, optional): the number of threads to use for cv2 operations.
                If None, doesn't change the value that cv2 uses. Defaults to 1.
        """

        self._data = data
        self._with_NaNs = with_NaNs

        # CHECK kernel
        self._kernel = self._check_kernel(kernel)

        # CV2 setup
        prev = None
        if cv2_threads is not None:
            prev = cv2.getNumThreads()
            cv2.setNumThreads(cv2_threads)

        # RUN
        self._sdev = self._sdev_loc()

        # CV2 reset
        if prev is not None: cv2.setNumThreads(prev)

    @property
    def sdev(self) ->  np.ndarray[tuple[int, ...], np.dtype[np.floating]]:
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
            return kernel
        elif isinstance(kernel, tuple):
            if len(kernel) != self._data.ndim:
                raise ValueError(
                    "If 'kernel' is given as a tuple, it must have the same number of elements "
                    "as 'data' has dimensions."
                )
            if not all(isinstance(k, int) for k in kernel):
                raise TypeError("All elements of 'kernel' tuple must be of type int.")
            return np.ones(kernel, dtype=np.float64)
        elif isinstance(kernel, int):
            return np.ones((kernel,) * self._data.ndim, dtype=np.float64)
        else:
            raise TypeError(
                "'kernel' must be an int, a tuple of ints, or a numpy ndarray."
            )

    def _sdev_loc(self) -> np.ndarray[tuple[int, ...], np.dtype[np.floating]]:
        """
        Computes the moving sample standard deviations. The size of each sample is defined by the
        kernel (square with a length of 'size').

        Returns:
            np.ndarray[tuple[int, ...], np.dtype[np.floating]]: Array of moving sample standard
                deviations.
        """

        if self._with_NaNs:
            # Create mask for valid (non-NaN) values
            valid_mask = ~np.isnan(self._data)
            data_filled = np.where(valid_mask, self._data, 0.0)

            # Compute mean and mean of squares with proper normalization
            sum_values = self._sliding_mean(data_filled)
            sum_squares = self._sliding_mean(data_filled ** 2)
            count = self._sliding_mean(valid_mask.astype(np.float64))

            # Avoid division by zero
            with np.errstate(divide='ignore', invalid='ignore'):
                mean = np.where(count > 0, sum_values / count, 0.0)
                mean_sq = np.where(count > 0, sum_squares / count, 0.0)

            # Compute variance
            variance = mean_sq - mean ** 2
            variance = np.maximum(variance, 0.0)  # Handle numerical errors
            return np.sqrt(variance)
        else:
            # STD
            mean2 = self._sliding_mean(self._data) ** 2
            variance = self._sliding_mean(self._data ** 2)
            variance -= mean2
            variance[variance <= 0] = 1e-20
            return np.sqrt(variance)

    def _sliding_mean(
            self,
            arr: np.ndarray[tuple[int, ...], np.dtype[Any]],
        ) -> np.ndarray[tuple[int, ...], np.dtype[Any]]:
        """
        Computes the sliding mean between the given 'arr' and the kernel.

        Args:
            arr (np.ndarray[tuple[int, ...], np.dtype[Any]]): the input array to convolve.

        Returns:
            np.ndarray[tuple[int, ...], np.dtype[Any]]: the result of the convolution.
        """

        output = np.empty(arr.shape, dtype=arr.dtype)

        if arr.ndim == 2:
            cv2.filter2D(
                arr,
                -1,  # Same pixel depth as input
                self._kernel,
                output,
                (-1, -1),  # Anchor is kernel center
                0,  # Optional offset
                cv2.BORDER_REFLECT,
            )
        elif arr.ndim == 3 and np.allclose(self._kernel, self._kernel.flat[0]):
            # APPROXIMATION (only works for uniform kernels)
            kernel_2d = np.ones((self._kernel.shape[1], self._kernel.shape[2]), dtype=np.float64)
            for i in range(arr.shape[0]):
                cv2.filter2D(
                    arr[i],
                    -1,
                    kernel_2d,
                    output[i],
                    (-1, -1),
                    0,
                    cv2.BORDER_REFLECT,
                )
            kernel_1d = np.ones((self._kernel.shape[0], 1), dtype=np.float64)
            for i in range(arr.shape[2]):
                dum = np.empty(output[:, :, i].shape, dtype=output.dtype)
                cv2.filter2D(
                    np.ascontiguousarray(output[:, :, i]),  # * contiguous for C implementation
                    -1,
                    kernel_1d,
                    dum,
                    (-1, -1),
                    0,
                    cv2.BORDER_REFLECT,
                )
                output[:, :, i] = dum
        else:
            convolve(
                arr,
                self._kernel,
                output=output,
                mode='mirror',
            )
        return output
