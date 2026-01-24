"""
To test a new implementation of the sigma clipping algorithm.
"""
from __future__ import annotations

# IMPORTs alias
import numpy as np

# IMPORTs sub
from numpy import ma
from numba import set_num_threads

# IMPORTs local
from programs.sigma_clipping.convolution import BorderType, Convolution
from programs.sigma_clipping.standard_deviation import FastStandardDeviation
from programs.sigma_clipping.sliding_mode.numba_functions import (
    tuple_sliding_nanmedian_3d, sliding_weighted_median_3d,
    tuple_sliding_nanmedian_nd, sliding_weighted_median_nd,
)

# TYPE ANNOTATIONs
from typing import cast, Literal, overload, Any
type KernelType = int | tuple[int, ...] | np.ndarray[tuple[int, ...], np.dtype[np.floating]]

# API public
__all__ = ["FastSigmaClipping"]

# ? should I make sure that the input array is of type float64 as the code wasn't working as
# ? intended (at least when the input wasn't of type floating).
# todo need to add .squeeze() when opening the FITS files.



class FastSigmaClipping[Output: np.ndarray | ma.MaskedArray]:
    """
    For a sliding sigma clip of an input array with a given kernel.
    Use the 'results' property to get the sigma clipped array.
    ! IMPORTANT: the kernel size must be odd and of the same dimensionality as the input array
    ! (when the kernel is given as an ndarray or a tuple of ints).

    Raises:
        ValueError: if the kernel size is even.
    """

    @overload
    def __init__(
            self: FastSigmaClipping[ma.MaskedArray],
            data: np.ndarray,
            kernel: KernelType = 3,
            center_choice: Literal['median', 'mean'] = 'median',
            sigma: float = 3.,
            sigma_lower: float | None = None,
            sigma_upper: float | None = None,
            max_iters: int | None = 5,
            borders: BorderType = 'reflect',
            threads: int | None = 1,
            masked_array: Literal[True] = True,
        ) -> None: ...

    @overload
    def __init__(
            self: FastSigmaClipping[np.ndarray],
            data: np.ndarray,
            kernel: KernelType = 3,
            center_choice: Literal['median', 'mean'] = 'median',
            sigma: float = 3.,
            sigma_lower: float | None = None,
            sigma_upper: float | None = None,
            max_iters: int | None = 5,
            borders: BorderType = 'reflect',
            threads: int | None = 1,
            *,
            masked_array: Literal[False],
        ) -> None: ...

    @overload  #fallback
    def __init__(
            self: FastSigmaClipping[np.ndarray | ma.MaskedArray],
            data: np.ndarray,
            kernel: KernelType = 3,
            center_choice: Literal['median', 'mean'] = 'median',
            sigma: float = 3.,
            sigma_lower: float | None = None,
            sigma_upper: float | None = None,
            max_iters: int | None = 5,
            borders: BorderType = 'reflect',
            threads: int | None = 1,
            masked_array: bool = True,
        ) -> None: ...

    def __init__(
            self,
            data: np.ndarray,
            kernel: KernelType = 3,
            center_choice: Literal['median', 'mean'] = 'median',
            sigma: float = 3.,
            sigma_lower: float | None = None,
            sigma_upper: float | None = None,
            max_iters: int | None = 5,
            borders: BorderType = 'reflect',
            threads: int | None = 1,
            masked_array: bool = True,
        ) -> None:
        """
        Runs the sliding sigma clipping where the flagged pixels are swapped with the center value
        (mean or median) for that pixel. The input data need to be have at least 2 dimensions and
        the kernel needs to be of the same dimensions than the input data.
        The result is accessed through the 'results' property and will be a numpy.ma.MaskedArray if
        'masked_array' is set to True, else a numpy.array. The sigma clipping is done iteratively
        'max_iters' number of times or till there are no more pixels flagged.
        NOTE: 
            ! kernel size must be odd (wouldn't make sense otherwise).
            ! kernel dimensionality must match data dimensionality (when kernel is not an int).

        Args:
            data (np.ndarray): the data to sigma clip.
            kernel (KernelType, optional): the kernel information used for computing the centers
                and standard deviations. Can be an int (square kernel), a tuple of ints (defining
                the shape of the kernel) or a numpy ndarray (defining the full kernel with
                weights). Defaults to 3.
            center_choice (Literal['median', 'mean'], optional): the function to use for computing
                the center value for each pixel. Defaults to 'median'.
            sigma (float): the number of standard deviations to use for both the lower and upper
                clipping limit. Overridden by 'sigma_lower' and/or 'sigma_upper'. 
            sigma_lower (float | None, optional): the number of standard deviations to use for
                the lower clipping limit. It will be set to 'sigma' if None. Defaults to None.
            sigma_upper (float | None, optional): the number of standard deviations to use for
                the upper clipping limit. It will be set to 'sigma' if None. Defaults to None.
            max_iters (int | None, optional): the maximum number of iterations to perform.
                If None, iterate until convergence. Defaults to 5.
            borders (BorderType, optional): the type of borders to use. These are the type of
                borders used by OpenCV (not all OpenCV borders are implemented as some don't have
                the equivalent in np.pad or scipy.ndimage). If None, uses adaptative borders, i.e.
                no padding and hence smaller kernels at the borders. Defaults to 'reflect'.
            threads (int | None, optional): the number of threads to use for numba and cv2
                parallelization. If None, uses the default number of threads. Defaults to 1.
                ! Might not work as expected given than numpy, numba and cv2 do not always let you
                ! set the number of threads at runtime.
            masked_array (bool, optional): whether to return a MaskedArray (True) or a normal
                ndarray (False). Defaults to True.
        """

        self._data = data
        self._sigma = sigma
        self._kernel = kernel
        self._borders = borders
        self._threads = threads
        self._masked_array = masked_array
        self._center_choice = center_choice
        self._sigma_lower = sigma_lower if sigma_lower is not None else sigma
        self._sigma_upper = sigma_upper if sigma_upper is not None else sigma
        self._max_iters = cast(int, max_iters if max_iters is not None else np.inf)

        # PADDING setup
        padding_settings = self._get_padding()
        self._padding_mode = padding_settings['mode']
        self._padding_reflect_type = padding_settings.get('reflect', 'even')
        self._padding_constant_values = padding_settings.get('constant_values', 0.)

        # CHECKs
        self._kernel_dim = self._check_kernel()
        self._check_data()

        # RUN
        if self._threads is not None: set_num_threads(self._threads)
        self._sigma_clipped = self._run()

    @property
    def results(self) -> Output:
        """
        The sigma clipped input where the pixels that where flagged are set to the center value
        (median or mean) gotten for that pixel.
        If 'masked_array' is True, the output is a MaskedArray where the masked pixels are the
        ones that were flagged during the sigma clipping process. Else, it's a normal ndarray.

        Returns:
            Output: the sigma clipped data.
        """
        return self._sigma_clipped

    def _get_padding(self) -> dict[str, Any]:
        """
        To get the corresponding np.pad parameters given the border type.
        """

        if self._borders is None:
            # ADAPTATIVE borders
            result = {
                'mode': 'constant',
                'constant_values': np.nan,
            }
        elif self._borders == 'reflect':
            result = {
                'mode': 'symmetric',
                'reflect': 'even',
            }
        elif self._borders == 'constant':
            result = {
                'mode': 'constant',
                'constant_values': 0.,
            }
        elif self._borders == 'replicate':
            result = {'mode': 'edge'}
        elif self._borders == 'wrap':
            result = {'mode': 'wrap'}
        else:
            raise ValueError(f"Unknown border type: {self._borders}")
        return result

    def _check_kernel(self) -> int:
        """
        Checks input kernel size validity.

        Raises:
            ValueError: if any kernel dimension is even.
            TypeError: if the kernel is not an int, tuple of ints or a numpy ndarray.
        """

        if isinstance(self._kernel, int):
            if self._kernel % 2 == 0: raise ValueError("Kernel size must be odd.")
            return self._data.ndim
        elif isinstance(self._kernel, tuple):
            if any(k % 2 == 0 for k in self._kernel):
                raise ValueError("All kernel dimensions must be odd.")
            return len(self._kernel)
        elif isinstance(self._kernel, np.ndarray):
            if any(s % 2 == 0 for s in self._kernel.shape):
                raise ValueError("All kernel dimensions must be odd.")
            return self._kernel.ndim
        else:
            raise TypeError("Kernel must be an int, tuple of ints or a numpy ndarray.")

    def _check_data(self) -> None:
        """
        Checks input data validity.

        Raises:
            ValueError: if the input data has less than 2 dimensions.
            ValueError: if the input data has not the same dimensionality as the kernel.
        """

        if self._data.ndim < 2:
            raise ValueError("Input data must have at least 2 dimensions.")
        elif self._data.ndim != self._kernel_dim:
            raise ValueError("Input data must have the same dimensionality as the kernel.")

    def _run(self) -> Output:
        """
        Runs the sigma clipping code where flagged pixels are set to the centers value gotten
        for that pixel (and after all iterations are done).
        The sigma clipping is done iteratively 'max_iters' number of times or till there are no
        more pixels that are flagged.
        If 'masked_array' is True, the output is a MaskedArray where the masked pixels are the
        ones that were flagged during the sigma clipping process.

        Returns:
            Output: the sigma clipped data.
        """

        output = self._data.copy()

        # TYPE CHECKER complains
        centers = np.empty(0, dtype=self._data.dtype)
        self._borders = cast(BorderType, self._borders)

        # COUNTs
        changed: bool = True
        iterations = 0

        # LOOP
        while (changed is True) and (iterations < self._max_iters):

            # CENTERs and STDDEVs
            centers = self._get_center(output)
            std_devs = FastStandardDeviation(
                data=output,
                kernel=self._kernel,
                borders=self._borders,
                with_NaNs=True,
                threads=self._threads,
            ).sdev

            diffs = output - centers
            lower = diffs < - self._sigma_lower * std_devs
            upper = diffs > self._sigma_upper * std_devs
            new_mask = lower | upper

            # UPDATE OUTPUT
            output[new_mask] = np.nan
            iterations += 1
            changed = bool(new_mask.any())

        # FILTER NaNs
        isnan = np.isnan(output)
        output[isnan] = centers[isnan]
        if self._masked_array: return cast(Output, ma.masked_array(output, mask=isnan))
        return cast(Output, output)

    def _get_center(self, data: np.ndarray) -> np.ndarray:
        """
        To get the sliding median of 'data' given a kernel size.

        Args:
            data (np.ndarray): the data to get the sliding median for.

        Returns:
            np.ndarray: the sliding median result.
        """

        if isinstance(self._kernel, (int, tuple)):
            return self._get_center_int_tuple(data)
        else:
            return self._get_center_custom(data)

    def _get_center_int_tuple(self, data: np.ndarray) -> np.ndarray:
        """
        To get the sliding mean of 'data' given a kernel size.
        The kernel value must be an int or a tuple of integers.

        Args:
            data (np.ndarray): the data to get the sliding mean for.

        Returns:
            np.ndarray: the sliding mean result.
        """

        # TYPE CHECKER complains
        kernel = cast(int | tuple[int, ...], self._kernel)  # for the type checker

        # KERNEL setup
        if isinstance(kernel, int): kernel = cast(tuple[int, ...], (kernel,) * self._kernel_dim)

        # MEAN
        if self._center_choice == 'mean': 
            return self._get_mean(data=data, kernel=np.ones(kernel, dtype=data.dtype))

        # MEDIAN
        pad = tuple((k // 2, k // 2) for k in kernel)
        padded = self._add_padding(data, pad)

        # MEDIAN choice
        if self._kernel_dim == 3: return tuple_sliding_nanmedian_3d(padded, kernel)#type:ignore
        return tuple_sliding_nanmedian_nd(padded, kernel)

    def _get_center_custom(self, data: np.ndarray) -> np.ndarray:
        """
        To get the sliding mean of 'data' given a kernel size.
        The kernel value must be a numpy ndarray.

        Args:
            data (np.ndarray): the data to get the sliding mean for.

        Returns:
            np.ndarray: the sliding mean result.
        """

        # TYPE CHECKER complains
        kernel = cast(np.ndarray, self._kernel)  # for the type checker

        # MEAN
        if self._center_choice == 'mean': return self._get_mean(data, kernel)

        # MEDIAN
        pad = tuple((k // 2, k // 2) for k in kernel.shape)
        padded = self._add_padding(data, pad)

        # MEDIAN choice
        # todo add an efficient 2D
        if self._kernel_dim == 3: return sliding_weighted_median_3d(padded, kernel)
        return sliding_weighted_median_nd(padded, kernel)

    def _get_mean(self, data: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        """
        Gives the sliding mean given the data and a kernel.

        Args:
            data (np.ndarray): the data to get the sliding mean for.
            kernel (np.ndarray): the kernel to use for the sliding mean.

        Returns:
            np.ndarray: the sliding mean result.
        """

        # TYPE CHECKER complains
        self._borders = cast(BorderType, self._borders)

        # NaN handling
        valid_mask = ~np.isnan(data)
        data_filled = np.where(valid_mask, data, 0.).astype(data.dtype)  # ? needed ?

        # SUM n MEAN
        sum_values = Convolution(
            data=data_filled,
            kernel=kernel,
            borders=self._borders,
            threads=self._threads,
        ).result
        count = Convolution(
            data=valid_mask.astype(data.dtype),
            kernel=np.ones(kernel.shape, dtype=data.dtype),  # no weights used here
            borders=self._borders,
            threads=self._threads,
        ).result
        with np.errstate(divide='ignore', invalid='ignore'):
            mean = np.where(count > 0, sum_values / count, 0.0)
        return mean

    def _add_padding(self, data: np.ndarray, pad: tuple[tuple[int, int], ...]) -> np.ndarray:
        """
        To add padding to 'data' given the pad widths and the padding mode.

        Args:
            data (np.ndarray): the data to pad.
            pad (tuple[tuple[int, int], ...]): the pad widths for each dimension.

        Returns:
            np.ndarray: the padded data.
        """

        if self._padding_mode == 'constant':
            padded = np.pad(
                array=data,
                pad_width=pad,
                mode=cast(Literal['edge'], self._padding_mode),
                constant_values=self._padding_constant_values,
            )
        elif self._padding_mode in ['reflect', 'symmetric']:  # ! cannot be reflect any more
            padded = np.pad(
                array=data,
                pad_width=pad,
                mode=cast(Literal['edge'], self._padding_mode),
                reflect_type=cast(Literal['even'], self._padding_reflect_type),
            )
        else:
            padded = np.pad(
                array=data,
                pad_width=pad,
                mode=cast(Literal['edge'], self._padding_mode),
            )
        return padded
