"""
Code to compute the moving sample standard deviation using full 3D convolution.
"""
from __future__ import annotations

# IMPORTs alias
import numpy as np

# IMPORTs sub
from scipy.ndimage import convolve

# IMPORTs personal
from common import Decorators

# TYPE ANNOTATIONs
from typing import Any

# API public
__all__ = ["Convolution3D"]



class Convolution3D:
    """
    Full direct 3D convolution implementation.
    """

    @Decorators.running_time
    def __init__(
            self,
            data: np.ndarray[tuple[int, ...], np.dtype[Any]],
            kernel_size: int,
            with_nans: bool = False,
        ) -> None:
        """
        Computes the moving sample standard deviation using a full 3D convolution directly.

        Args:
            data (np.ndarray[tuple[int, ...], np.dtype[Any]]): the data for which to compute the
                moving sample standard deviation.
            kernel_size (int): the size of the moving kernel.
            with_nans (bool, optional): whether to handle NaNs in the data. Defaults to False.
        """

        self._data = data
        self._kernel_size = kernel_size
        self._with_nans = with_nans

        # RUN
        self._std = self._run()

    @property
    def sdev(self) -> np.ndarray[tuple[int, ...], np.dtype[np.floating]]:
        """
        Gives the computed moving sample standard deviation.

        Returns:
            np.ndarray[tuple[int, ...], np.dtype[np.floating]]: the moving sample standard
                deviation.
        """
        return self._std

    def _run(self) -> np.ndarray[tuple[int, ...], np.dtype[np.floating]]:
        """
        Runs the 3D convolution to get the moving sample standard deviation.

        Returns:
            np.ndarray[tuple[int, ...], np.dtype[np.floating]]: the moving sample standard
                deviation.
        """

        # Create normalized kernel
        kernel_shape = (self._kernel_size,) * self._data.ndim
        kernel = np.ones(kernel_shape) / (self._kernel_size ** self._data.ndim)

        if self._with_nans:
            data_filled = np.where(np.isnan(self._data), 0.0, self._data)
            valid_mask = ~np.isnan(self._data)
            
            # Convolve with reflect mode (matches generic_filter default)
            mean = convolve(data_filled.astype(np.float64), kernel, mode='reflect')
            mean_sq = convolve((data_filled ** 2).astype(np.float64), kernel, mode='reflect')
            count = convolve(valid_mask.astype(np.float64), kernel, mode='reflect')
            
            with np.errstate(divide='ignore', invalid='ignore'):
                mean = np.where(count > 0, mean / count, 0.0)
                mean_sq = np.where(count > 0, mean_sq / count, 0.0)
        else:
            # Use reflect mode to match generic_filter
            mean = convolve(self._data.astype(np.float64), kernel, mode='reflect')
            mean_sq = convolve((self._data ** 2).astype(np.float64), kernel, mode='reflect')

        variance = mean_sq - mean ** 2
        variance = np.maximum(variance, 0.0)
        return np.sqrt(variance).astype(self._data.dtype)
