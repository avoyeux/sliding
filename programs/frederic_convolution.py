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
__all__ = ["ConvolutionSTD"]



class ConvolutionSTD:
    """
    To compute the moving sample standard deviations using convolutions.
    """

    @Decorators.running_time
    def __init__(
            self,
            data: np.ndarray[tuple[int, ...], np.dtype[Any]],
            kernel_size: int,
        ) -> None:
        """
        Computes the moving sample standard deviations using convolutions. The size of each sample
        is defined by the kernel (square with a length of 'kernel_size').
        To retrieve the computed standard deviations, use the 'sdev' property.

        Args:
            data (np.ndarray[tuple[int, ...], np.dtype[Any]]): the data for which the moving sample
                standard deviations are computed.
            kernel_size (int): the size of the kernel (square) used for computing the moving sample
                standard deviations.
            verbose (int, optional): verbosity level for the prints. Defaults to 0.
            flush (bool, optional): whether to flush the prints. Defaults to False.
        """

        self._data = data
        self._kernel = np.ones((kernel_size,) * 2, dtype=np.float64) / (kernel_size ** 2)

        # RUN
        self._sdev = self._sdev_loc()

    @property
    def sdev(self) ->  np.ndarray[tuple[int, ...], np.dtype[np.floating]]:
        """
        Returns the moving sample standard deviations.

        Returns:
            np.ndarray[tuple[int, ...], np.dtype[np.floating]]: Array of moving sample standard
                deviations.
        """
        return self._sdev

    def _sdev_loc(self) -> np.ndarray[tuple[int, ...], np.dtype[np.floating]]:
        """
        Computes the moving sample standard deviations. The size of each sample is defined by the
        kernel (square with a length of 'size').

        Returns:
            np.ndarray[tuple[int, ...], np.dtype[np.floating]]: Array of moving sample standard
                deviations.
        """

        # STD
        mean2 = self._convolution(self._data) ** 2
        variance = self._convolution(self._data ** 2)
        variance -= mean2
        variance[variance <= 0] = 1e-20
        return np.sqrt(variance)

    def _convolution(
            self,
            arr: np.ndarray[tuple[int, ...], np.dtype[Any]],
        ) -> np.ndarray[tuple[int, ...], np.dtype[Any]]:
        """
        Does a convolution between the given 'arr' and the kernel.

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
        elif arr.ndim == 3:
            for i in range(arr.shape[0]):
                cv2.filter2D(
                    arr[i],
                    -1,  # Same pixel depth as input
                    self._kernel,
                    output[i],
                    (-1, -1),  # Anchor is kernel center
                    0,  # Optional offset
                    cv2.BORDER_REFLECT,
                )
            kernel_1d = (
                np.ones((self._kernel.shape[0], 1), dtype=np.float64) / self._kernel.shape[0]
            )
            for i in range(arr.shape[2]):
                dum = np.empty_like(output[:, :, i])
                cv2.filter2D(
                    np.ascontiguousarray(output[:, :, i]),  # * contiguous for C implementation
                    -1,  # Same pixel depth as input
                    kernel_1d,
                    dum,
                    (-1, -1),  # Anchor is kernel center
                    0,  # Optional offset
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
