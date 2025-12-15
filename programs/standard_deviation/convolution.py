"""
Code to calculate the sliding mean using a convolution.
This code is Dr. Auchere's implementation of the sliding mean.
"""
from __future__ import annotations

# IMPORTs third-party
import cv2

# IMPORTs alias
import numpy as np

# IMPORTs sub
from scipy.ndimage import convolve

# API public
__all__ = ["Convolution"]



class Convolution[Data: np.ndarray[tuple[int, ...], np.dtype[np.floating]]]:
    """
    To compute the convolution between an array and a kernel.
    No NaN handling is done.
    """

    def __init__(
            self,
            data: Data,
            kernel: Data,
            cv2_threads: int | None = 1,
        ) -> None:
        """
        Computes the convolution between the given array and the kernel.
        To access the results, use the 'result' property.
        No NaN handling is done.

        Args:
            data (Data): the data to convolve.
            kernel (Data: the kernel to use for the convolution.
            cv2_threads (int | None, optional): the number of threads to use for cv2 operations.
                If None, doesn't change the value that cv2 uses. Defaults to 1.
        """

        self._input_dtype = data.dtype
        self._data = data.copy()
        self._kernel = kernel

        # CV2 setup
        prev = None
        if cv2_threads is not None:
            prev = cv2.getNumThreads()
            cv2.setNumThreads(cv2_threads)

        # RUN
        self.run()

        # CV2 reset
        if prev is not None: cv2.setNumThreads(prev)

    @property
    def result(self) -> Data:
        """
        Gives the result of the convolution between the data and the kernel.

        Returns:
            Data: the result of the convolution. Has the same shape and type as the input array.
        """

        if self._data.dtype != self._input_dtype:
            return self._data.astype(self._input_dtype, copy=False)#type:ignore
        return self._data

    def run(self) -> None:
        """
        Does the convolution with the given kernel.
        An approximation is applied when possible.
        """

        if self._data.ndim == 2:
            cv2.filter2D(
                self._data,
                -1,  # Same pixel depth as input
                self._kernel,
                self._data,
                (-1, -1),  # Anchor is kernel center
                0,  # Optional offset
                cv2.BORDER_REFLECT,
            )
        elif self._data.ndim == 3 and np.allclose(self._kernel, self._kernel.flat[0]):
            # APPROXIMATION (only works for uniform kernels)
            kernel_2d = np.ones((self._kernel.shape[1], self._kernel.shape[2]), dtype=np.float64)
            for i in range(self._data.shape[0]):
                cv2.filter2D(
                    self._data[i],
                    -1,
                    kernel_2d,
                    self._data[i],
                    (-1, -1),
                    0,
                    cv2.BORDER_REFLECT,
                )
            kernel_1d = np.ones((self._kernel.shape[0], 1), dtype=np.float64)
            for i in range(self._data.shape[2]):
                dum = np.empty(self._data[:, :, i].shape, dtype=self._data.dtype)
                cv2.filter2D(
                    np.ascontiguousarray(self._data[:, :, i]),  # * contiguous for C implementation
                    -1,
                    kernel_1d,
                    dum,
                    (-1, -1),
                    0,
                    cv2.BORDER_REFLECT,
                )
                self._data[:, :, i] = dum
        else:
            convolve(
                self._data,
                self._kernel,
                output=self._data,
                mode='mirror',
            )
