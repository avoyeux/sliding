"""
Another possible way of doing so. Similar to Dr. Auchere's approach
"""

# IMPORTs alias
import numpy as np

# IMPORTs sub
from scipy.ndimage import convolve1d

# IMPORTs personal
from common import Decorators

# TYPE ANNOTATIONs
from typing import Any, Callable

# API public
__all__ = ["Convolution1D"]



class Convolution1D:
    """
    To compute the moving sample standard deviation using 1D convolutions.
    """

    @Decorators.running_time
    def __init__(
            self,
            data: np.ndarray[tuple[int, ...], np.dtype[Any]],
            kernel_size: int,
            with_nans: bool = False,
        ) -> None:
        """
        Computes the moving sample standard deviation by using 1D convolutions.
        It is basically the same way of thinking than Dr. Auchere's code, with the exception
        that the border conditions weren't taken into account here.

        Args:
            data (np.ndarray[tuple[int, ...], np.dtype[Any]]): the data for which the moving
                sample standard deviation is needed.
            kernel_size (int): the kernel size that defines each sample.
            with_nans (bool, optional): whether to handle NaNs in the data. Defaults to False.
        """

        self._data = data
        self._kernel_1d = np.ones(kernel_size) / kernel_size
        self._with_nans = with_nans

        # RUN
        self._std = self._run()

    @property
    def sdev(self) ->  np.ndarray[tuple[int, ...], np.dtype[np.floating]]:
        """
        Returns the moving sample standard deviation.

        Returns:
            np.ndarray[tuple[int, ...], np.dtype[np.floating]]: the moving sample standard
                deviation.
        """

        return self._std


    def _run(self) -> np.ndarray[tuple[int, ...], np.dtype[np.floating]]:
        """
        Runs the computation of the moving standard deviation

        Returns:
            np.ndarray[tuple[int, ...], np.dtype[np.floating]]: _description_
        """

        if self._with_nans:
            data_filled = np.nan_to_num(self._data, nan=0.0)
            valid_mask = ~np.isnan(self._data)
            
            # Convolve along each axis separately
            mean = data_filled.copy()
            mean_sq = (data_filled ** 2).copy()
            count = valid_mask.astype(float).copy()

            for axis in range(self._data.ndim):
                mean = convolve1d(mean, self._kernel_1d, axis=axis, mode='constant', cval=0.0)
                mean_sq = convolve1d(mean_sq, self._kernel_1d, axis=axis, mode='constant', cval=0.0)
                count = convolve1d(count, self._kernel_1d, axis=axis, mode='constant', cval=0.0)

            mean = mean / np.maximum(count, 1e-10)
            mean_sq = mean_sq / np.maximum(count, 1e-10)
        else:
            mean = self._data.copy()
            mean_sq = (self._data ** 2).copy()
            
            for axis in range(self._data.ndim):
                mean = convolve1d(mean, self._kernel_1d, axis=axis, mode='constant', cval=0.0)
                mean_sq = convolve1d(mean_sq, self._kernel_1d, axis=axis, mode='constant', cval=0.0)

        variance = mean_sq - mean ** 2
        variance = np.maximum(variance, 0)

        return np.sqrt(variance)

    def _std_func(self) -> Callable[[np.ndarray[tuple[int, ...], np.dtype[Any]]], np.floating]:
        """
        To choose the right standard deviation function depending on the instance arguments.
        The choice is between np.std and np.nanstd.

        Returns:
            Callable: the standard deviation function to use.
        """
        return np.nanstd if self._with_nans else np.std
