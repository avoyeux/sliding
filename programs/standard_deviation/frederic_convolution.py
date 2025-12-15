"""
Contains the code that needs testing to see if it can be used for SPICE (for moving sample
standard deviation calculations).
"""
from __future__ import annotations

# IMPORTs alias
import numpy as np

# IMPORTs local
from programs.standard_deviation.convolution import Convolution

# IMPORTs personal
from common import Decorators

# API public
__all__ = ["QuickSTDs"]



class QuickSTDs[Data: np.ndarray[tuple[int, ...], np.dtype[np.floating]]]:
    """
    To compute the moving sample standard deviations using convolutions.
    """

    @Decorators.running_time
    def __init__(
            self,
            data: Data,
            kernel: int | tuple[int, ...] | Data,
            with_NaNs: bool = True,
            cv2_threads: int | None = 1,
        ) -> None:
        """
        todo update docstring
        Computes the moving sample standard deviations using convolutions. The size of each sample
        is defined by the kernel. If you decide to choose to have different weights in the kernel,
        keep in mind that the standard deviation will take a little longer to compute (no
        approximation possible).
        To retrieve the computed standard deviations, use the 'sdev' property.

        Args:
            data (np.ndarray[tuple[int, ...], np.dtype[np.floating]]): the data for which the
                moving sample standard deviations are computed.
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
        self._cv2_threads = cv2_threads

        # CHECK kernel
        self._kernel = self._check_kernel(kernel)

        # RUN
        self._sdev = self._sdev_loc()

    @property
    def sdev(self) ->  Data:
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
            return np.ones(kernel, dtype=np.float64) / np.prod(kernel)
        elif isinstance(kernel, int):
            normalised = (
                np.ones((kernel,) * self._data.ndim, dtype=np.float64) /
                (kernel ** self._data.ndim)
            )
            return normalised
        else:
            raise TypeError(
                "'kernel' must be an int, a tuple of ints, or a numpy ndarray."
            )

    def _sdev_loc(self) -> Data:
        """
        Computes the moving sample standard deviations. The size of each sample is defined by the
        kernel (square with a length of 'size').

        Returns:
            np.ndarray[tuple[int, ...], np.dtype[np.floating]]: Array of moving sample standard
                deviations.
        """

        if self._with_NaNs:
            # VALID (non-NaN)
            valid_mask = ~np.isnan(self._data)
            data_filled = np.where(valid_mask, self._data, 0.)

            # SUM n MEAN
            sum_values = Convolution(data_filled, self._kernel, self._cv2_threads).result
            sum_squares = Convolution(data_filled ** 2, self._kernel, self._cv2_threads).result
            count = Convolution(
                data=valid_mask.astype(np.float64),
                kernel=self._kernel,
                cv2_threads=self._cv2_threads,
            ).result

            with np.errstate(divide='ignore', invalid='ignore'):
                mean = np.where(count > 0, sum_values / count, 0.0)
                mean_sq = np.where(count > 0, sum_squares / count, 0.0)

            # STD
            variance = mean_sq - mean ** 2
            variance = np.maximum(variance, 0.0)
            return np.sqrt(variance)
        else:
            # STD
            mean2 = Convolution(self._data, self._kernel, self._cv2_threads).result ** 2
            variance = Convolution(self._data ** 2, self._kernel, self._cv2_threads).result
            variance -= mean2
            variance[variance <= 0] = 1e-20
            return np.sqrt(variance)
