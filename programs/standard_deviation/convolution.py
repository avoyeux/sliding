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
from threadpoolctl import threadpool_limits

# TYPE ANNOTATIONs
from typing import Literal
type BorderType = Literal['reflect', 'constant', 'replicate', 'wrap'] | None

# API public
__all__ = ["Convolution"]



class Convolution[Data: np.ndarray[tuple[int, ...], np.dtype[np.floating]]]:
    """
    To compute the convolution between an array and a kernel.
    No NaN handling is done.
    """

    BORDER_DICT = {
        'reflect': cv2.BORDER_REFLECT,
        'constant': cv2.BORDER_CONSTANT,
        'replicate': cv2.BORDER_REPLICATE,
        'wrap': cv2.BORDER_WRAP,
    }

    def __init__(
            self,
            data: Data,
            kernel: Data,
            borders: BorderType = 'reflect',
            threads: int | None = 1,
        ) -> None:
        """
        Computes the convolution between the given array and the kernel.
        To access the results, use the 'result' property.
        No NaN handling is done.

        Args:
            data (Data): the data to convolve.
            kernel (Data: the kernel to use for the convolution.
            borders (BorderType, optional): the type of borders to use. These are the type of
                borders used by OpenCV (not all OpenCV borders are implemented as some don't
                have the equivalent in np.pad or scipy.ndimage). If None, uses adaptative borders,
                i.e. no padding and hence smaller kernels at the borders. Defaults to 'reflect'.
            threads (int | None, optional): the number of threads to use for the computation.
                If None, doesn't change change the default behaviour. Defaults to 1.
        """

        self._borders = borders
        self._input_dtype = data.dtype
        self._data: Data = data.copy()
        self._kernel = kernel

        # RUN
        if threads is not None:
            with threadpool_limits(limits=threads): self._run()

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

    def _run(self) -> None:
        """
        Runs the convolution with the specified border handling.
        """

        if self._borders is None:
            # ADAPTATIVE kernel for the borders
            self._convolve_normalised()
        else:
            # STANDARD borders
            self._convolve(self._data, borders=self._borders)

    def _convolve_normalised(self) -> None:
        """
        Does the convolution in the case where no borders are wanted (i.e. adaptative kernels).
        This is different than the usual as you have to manually calculate the values at the
        borders doing a double convolution.
        """

        # APPROXIMATION 3D uniform kernel
        if self._data.ndim == 3 and np.allclose(self._kernel, self._kernel.flat[0]):
            # KERNEL 2D
            kernel_2d = np.ones(
                (self._kernel.shape[1], self._kernel.shape[2]),
                dtype=self._data.dtype,
            ) * self._kernel.flat[0]

            # CONVOLUTION 2D
            for i in range(self._data.shape[0]):
                cv2.filter2D(
                    self._data[i],
                    -1,
                    kernel_2d,
                    self._data[i],
                    (-1, -1),
                    0,
                    cv2.BORDER_CONSTANT,
                )

            # NORMALISATION 2D
            ones_2d = np.ones((self._data.shape[1], self._data.shape[2]), dtype=self._data.dtype)
            counts_2d = np.empty(ones_2d.shape, dtype=self._data.dtype)
            cv2.filter2D(ones_2d, -1, kernel_2d, counts_2d, (-1, -1), 0, cv2.BORDER_CONSTANT)
            self._data /= counts_2d[None, ...]#type:ignore

            # KERNEL 1D
            kernel_1d = np.ones((self._kernel.shape[0], 1), dtype=self._data.dtype)

            # CONVOLUTION 1D
            for i in range(self._data.shape[2]):
                dum = np.empty(self._data[:, :, i].shape, dtype=self._data.dtype)
                cv2.filter2D(
                    np.ascontiguousarray(self._data[:, :, i]),
                    -1,
                    kernel_1d,
                    dum,
                    (-1, -1),
                    0,
                    cv2.BORDER_CONSTANT,
                )
                self._data[:, :, i] = dum

            # NORMALISATION 1D
            ones_1d = np.ones((self._data.shape[0], self._data.shape[1]), dtype=self._data.dtype)
            counts_1d = np.empty(ones_1d.shape, dtype=self._data.dtype)
            cv2.filter2D(
                np.ascontiguousarray(ones_1d),
                -1,
                kernel_1d,
                counts_1d,
                (-1, -1),
                0,
                cv2.BORDER_CONSTANT,
            )
            self._data /= counts_1d[..., None]#type:ignore

        else:
            # CONVOLUTION standard
            self._convolve(self._data, borders='constant')
            counts = np.empty(self._data.shape, dtype=self._data.dtype)
            self._convolve(
                data=np.ones(self._data.shape, dtype=self._data.dtype),#type:ignore
                borders='constant',
                output=counts,#type:ignore
            )

            # NORMALISATION
            self._data /= counts#type:ignore

    def _convolve(self, data: Data, borders: str, output: Data | None = None) -> None:
        """
        Does the convolution of the data with a given kernel and border choice.
        If 'output' is None, 'data' is changed in place.

        Args:
            data (Data): the data to convolve.
            borders (str): the type of borders to use.
            output (Data | None, optional): the array to store the result. If None, modifies 'data'
                in place. Defaults to None.
        """

        if output is None: output = data

        if data.ndim == 2:
            cv2.filter2D(
                data,
                -1,  # Same pixel depth as input
                self._kernel,
                output,
                (-1, -1),  # Anchor is kernel center
                0,  # Optional offset
                self.BORDER_DICT[borders],
            )
        elif data.ndim == 3 and np.allclose(self._kernel, self._kernel.flat[0]):
            # APPROXIMATION (only works for uniform kernels)
            kernel_2d = np.ones(
                (self._kernel.shape[1], self._kernel.shape[2]),
                dtype=data.dtype,
            ) * self._kernel.flat[0]
            for i in range(data.shape[0]):
                cv2.filter2D(
                    data[i],
                    -1,
                    kernel_2d,
                    output[i],
                    (-1, -1),
                    0,
                    self.BORDER_DICT[borders],
                )
            kernel_1d = np.ones((self._kernel.shape[0], 1), dtype=data.dtype)
            for i in range(data.shape[2]):
                dum = np.empty(data[:, :, i].shape, dtype=data.dtype)
                cv2.filter2D(
                    np.ascontiguousarray(data[:, :, i]),  # * contiguous for C implementation
                    -1,
                    kernel_1d,
                    dum,
                    (-1, -1),
                    0,
                    self.BORDER_DICT[borders],
                )
                output[:, :, i] = dum
        else:
            convolve(
                data,
                self._kernel,
                output=output,
                mode=borders if borders != 'replicate' else 'nearest',
                cval=0.,
            )
