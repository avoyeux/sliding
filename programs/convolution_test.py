"""
Contains the code that is being test to see if it can be used for SPICE.
"""
from __future__ import annotations

# IMPORTs standard
import cv2

# IMPORTs alias
import numpy as np

# IMPORTs sub
from scipy.ndimage import convolve

# IMPORTs local
from common import Decorators

# TYPE ANNOTATIONs
from typing import Any

# API public
__all__ = ["ConvolutionSTD"]

# todo add the actual sigma clipping method.



class ConvolutionSTD:
    """
    To compute the moving sample standard deviations using convolutions.
    """

    @Decorators.running_time
    def __init__(
            self,
            image: np.ndarray[tuple[int, ...], np.dtype[Any]],
            kernel_size: int,
            verbose: int = 0, 
            flush: bool = False,
        ) -> None:
        """
        Computes the moving sample standard deviations using convolutions. The size of each sample
        is defined by the kernel (square with a length of 'kernel_size').
        To retrieve the computed standard deviations, use the 'sdev' property.

        Args:
            image (np.ndarray[tuple[int, ...], np.dtype[Any]]): _description_
            kernel_size (int): _description_
            verbose (int, optional): _description_. Defaults to 0.
            flush (bool, optional): _description_. Defaults to False.
        """

        # CONFIG attributes
        self._verbose = verbose
        self._flush = flush

        self._image = image
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
        mean2 = self._convolution(self._image) ** 2
        variance = self._convolution(self._image ** 2)
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
            for i in range(arr.shape[2]):
                dum = np.empty_like(output[:, :, i])
                cv2.filter2D(
                    np.copy(output[:, :, i]),
                    -1,  # Same pixel depth as input
                    np.expand_dims(self._kernel, axis=1),
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
