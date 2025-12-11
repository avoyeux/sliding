"""
Contains the moving sample standard deviation that is used right now for SPICE.
"""
from __future__ import annotations

# IMPORTs alias
import numpy as np

# IMPORTs sub
from scipy.ndimage import generic_filter

# IMPORTs personal
from common import Decorators

# TYPE ANNOTATIONs
from typing import Any, Callable

# API public
__all__ = ["GenericFilter"]



class GenericFilter:
    """
    Simple class that just implements scipy.ndimage.generic_filter.
    """

    @Decorators.running_time
    def __init__(
            self,
            data: np.ndarray[tuple[int, ...], np.dtype[Any]],
            kernel_size: int,
            with_nans: bool = False,
            verbose: int = 0,
            flush: bool = False,
        ) -> None:
        """
        Computes the moving sample standard deviation for a given data.
        To get the result, use the 'sdev' property.

        Args:
            data (np.ndarray[tuple[int, ...], np.dtype[Any]]): the data for which to compute the
                standard deviation.
            kernel_size (int): the kernel size that defines a sample.
            with_nans (bool, optional): whether the data contains NaN values. Defaults to False.
            verbose (int, optional): verbosity level for the prints. Defaults to 0.
            flush (bool, optional): whether to flush the prints. Defaults to False.
        """

        # CONFIG attributes
        self._verbose = verbose
        self._flush = flush

        self._data = data
        self._kernel_size = (kernel_size,) * data.ndim
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
        Runs the moving sample standard deviation.

        Returns:
            np.ndarray[tuple[int, ...], np.dtype[np.floating]]: _description_
        """

        result = generic_filter(
            self._data.copy(),
            self._std_func(),
            self._kernel_size,
        )
        return result

    def _std_func(self) -> Callable[[np.ndarray[tuple[int, ...], np.dtype[Any]]], np.floating]:
        """
        To choose the right standard deviation function depending on the instance arguments.
        The choice is between np.std and np.nanstd.

        Returns:
            Callable: the standard deviation function to use.
        """
        return np.nanstd if self._with_nans else np.std
