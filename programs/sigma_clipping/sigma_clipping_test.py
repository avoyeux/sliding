"""
To test a new implementation of the sigma clipping algorithm.
"""
from __future__ import annotations

# IMPORTs alias
import numpy as np

# IMPORTs sub
from numpy import ma
from numba import njit, prange, set_num_threads

# IMPORTs local
from programs.standard_deviation import QuickSTDs

# IMPORTs personal
from common import Decorators

# TYPE ANNOTATIONs
from typing import cast, Literal, overload
type PadModeType = Literal[
    'constant', 'edge', 'linear_ramp', 'maximum', 'mean', 'median', 'minimum', 'reflect',
    'symmetric', 'wrap', 'empty',
]

# API public
__all__ = ["FastSigmaClipping"]



@njit
def _fast_median(window: np.ndarray) -> np.floating | float:
    """
    To get the median of an odd sized kernel using partitioning.
    Also takes into account NaN values.

    Args:
        window (np.ndarray): the window to get the median value.

    Returns:
        np.floating | float: the median value of the window. If the window is empty, returns
            NaN.
    """

    valid = window[~np.isnan(window)]
    n = valid.size
    if n == 0: return np.nan

    # ODD
    if n % 2 == 1: return np.partition(valid, n // 2)[n // 2]

    # EVEN
    partitioned = np.partition(valid, [n // 2 - 1, n // 2])
    return 0.5 * (partitioned[n // 2 - 1] + partitioned[n // 2])

@njit(parallel=True)
def _sliding_nanmedian_3d(data: np.ndarray, size: int) -> np.ndarray:
    """
    To get the sliding median value for a given cubic kernel size. Keep in mind that the input
    data must be pre-padded to handle borders correctly.
    This is done using numba as I didn't find any other way to efficiently get the sliding
    median while there are NaN values inside the data.

    Args:
        data (np.ndarray): the padded data to get the sliding median for. Can and should contain
            NaNs.
        size (int): the size of the cubic kernel.

    Returns:
        np.ndarray: the sliding median result.
    """

    depth, rows, cols = data.shape
    pad  = size // 2

    results = np.empty((depth - 2 * pad, rows - 2 * pad, cols - 2 * pad), dtype=data.dtype)

    for j in prange(cols - 2 * pad):  # as cols is 128 but depth is 36 and rows 1024
        for i in range(rows - 2 * pad):
            for d in range(depth - 2 * pad):

                window = data[d:d + size, i:i + size, j:j + size].ravel()
                results[d, i, j] = _fast_median(window)
    return results

@njit(parallel=True)
def _sliding_nanmean_3d(data: np.ndarray, size: int) -> np.ndarray:
    """
    To get the sliding mean value for a given cubic kernel size. Keep in mind that the input data
    must be pre-padded to handle borders correctly.

    Args:
        data (np.ndarray): the padded data to get the sliding mean for. Can and should contain
            NaNs.
        size (int): the size of the cubic kernel.

    Returns:
        np.ndarray: the sliding mean result.
    """

    depth, rows, cols = data.shape
    pad  = size // 2

    results = np.empty((depth - 2 * pad, rows - 2 * pad, cols - 2 * pad), dtype=data.dtype)

    for j in prange(cols - 2 * pad):  # as cols is 128 but depth is 36 and rows 1024
        for i in range(rows - 2 * pad):
            for d in range(depth - 2 * pad):

                window = data[d:d + size, i:i + size, j:j + size].ravel()
                valid = window[~np.isnan(window)]
                n = valid.size
                if n == 0:
                    results[d, i, j] = np.nan
                else:
                    results[d, i, j] = np.mean(valid)
    return results


class FastSigmaClipping[Output: np.ndarray | ma.MaskedArray]:
    """
    To sigma clip an input 3 dimensional array with a kernel size.
    Use the 'results' property to get the sigma clipped array.

    Raises:
        ValueError: if the kernel size is even.
    """

    @overload
    def __init__(
            self: FastSigmaClipping[ma.MaskedArray],
            data: np.ndarray,
            size: int,
            center_choice: Literal['median', 'mean'] = 'median',
            sigma: float = 3.,
            sigma_lower: float | None = None,
            sigma_upper: float | None = None,
            max_iters: int | None = 5,
            masked_array: Literal[True] = True,
            padding_mode: PadModeType = 'symmetric',
            padding_reflect_type: Literal['even', 'odd'] = 'even',
            padding_constant_values: float = np.nan,
            numba_threads: int | None = None,
        ) -> None: ...

    @overload
    def __init__(
            self: FastSigmaClipping[np.ndarray],
            data: np.ndarray,
            size: int,
            center_choice: Literal['median', 'mean'] = 'median',
            sigma: float = 3.,
            sigma_lower: float | None = None,
            sigma_upper: float | None = None,
            max_iters: int | None = 5,
            *,
            masked_array: Literal[False],
            padding_mode: PadModeType = 'symmetric',
            padding_reflect_type: Literal['even', 'odd'] = 'even',
            padding_constant_values: float = np.nan,
            numba_threads: int | None = None,
        ) -> None: ...

    @overload  #fallback
    def __init__(
            self: FastSigmaClipping[np.ndarray | ma.MaskedArray],
            data: np.ndarray,
            size: int,
            center_choice: Literal['median', 'mean'] = 'median',
            sigma: float = 3.,
            sigma_lower: float | None = None,
            sigma_upper: float | None = None,
            max_iters: int | None = 5,
            masked_array: bool = True,
            padding_mode: PadModeType = 'symmetric',
            padding_reflect_type: Literal['even', 'odd'] = 'even',
            padding_constant_values: float = np.nan,
            numba_threads: int | None = None,
        ) -> None: ...

    @Decorators.running_time
    def __init__(
            self,
            data: np.ndarray,
            size: int,
            center_choice: Literal['median', 'mean'] = 'median',
            sigma: float = 3.,
            sigma_lower: float | None = None,
            sigma_upper: float | None = None,
            max_iters: int | None = 5,
            masked_array: bool = True,
            padding_mode: PadModeType = 'symmetric',
            padding_reflect_type: Literal['even', 'odd'] = 'even',
            padding_constant_values: float = np.nan,
            numba_threads: int | None = None,
        ) -> None:
        """
        Runs the sigma clipping where the flagged pixels are swapped with the center value (mean or
        median) for that pixel. The  input data is assumed to be 3 dimensional.
        The result can be gotten from the 'results' property and will be an numpy.ma.MaskedArray if
        'masked_array' is set to True, else a numpy.array.
        The sigma clipping is done iteratively 'max_iters' number of times or till there are no
        more pixels that are flagged.
        NOTE: 
            ! kernel size must be odd (wouldn't make sense otherwise).
            use default padding mode values to replicate scipy's reflect behaviour.

        Args:
            data (np.ndarray): the data to sigma clip.
            size (int): the size of the kernel (cubic) used for computing the centers and standard
                deviations.
            center_choice (Literal['median', 'mean'], optional): the function to use for computing
                the center value for each pixel. Defaults to 'median'.
            sigma (float): the number of standard deviations to use for both the lower and upper
                clipping limit. overridden by 'sigma_lower' and 'sigma_upper'. 
            sigma_lower (float | None, optional): the number of standard deviations to use for
                the lower clipping limit. It will be set to 'sigma' if None. Defaults to None.
            sigma_upper (float | None, optional): the number of standard deviations to use for
                the upper clipping limit. It will be set to 'sigma' if None. Defaults to None.
            max_iters (int | None, optional): the maximum number of iterations to perform.
                If None, iterate until convergence. Defaults to 5.
            masked_array (bool, optional): whether to return a MaskedArray (True) or a normal
                ndarray (False). Defaults to True.
            padding_mode (PadModeType, optional): the padding mode of numpy.pad to use when
                padding the input data for center calculations. If no padding is desired (hence
                smaller windows on the edges), set it to 'constant' and 'padding_constant_values'
                to np.nan. Defaults to 'symmetric'.
            padding_reflect_type (Literal['even', 'odd'], optional): the reflect type to use when
                'padding_mode' is set to 'reflect' or 'symmetric'. Defaults to 'even'.
            padding_constant_values (float, optional): the constant value to use when
                'padding_mode' is set to 'constant'. Defaults to np.nan.
            numba_threads (int | None, optional): the number of threads to use for numba
                parallelization. If None, uses the default number of threads. Defaults to None.

        Raises:
            ValueError: if the kernel size is not odd.
        """

        # CHECK odd kernel size
        if size % 2 == 0: raise ValueError("Kernel size must be odd.")

        self._data = data
        self._size = size
        self._sigma = sigma
        self._masked_array = masked_array
        self._center_choice = center_choice
        self._sigma_lower = sigma_lower if sigma_lower is not None else sigma
        self._sigma_upper = sigma_upper if sigma_upper is not None else sigma
        self._max_iters = cast(int, max_iters if max_iters is not None else np.inf)
        self._padding_mode = padding_mode
        self._padding_reflect_type = padding_reflect_type
        self._padding_constant_values = padding_constant_values

        # RUN
        if numba_threads is not None: set_num_threads(numba_threads)
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

        # PLACEHOLDER so IDE doesn't complain
        centers = np.empty(1)

        # COUNTs
        changed: bool = True
        iterations = 0

        # LOOP
        while (changed is True) and (iterations < self._max_iters):

            # CENTERs and STDDEVs
            centers = self._get_center(output)
            std_devs = QuickSTDs(data=output, kernel_size=self._size, with_NaNs=True).sdev

            diffs = output - centers
            lower = diffs < - self._sigma_lower * std_devs
            upper = diffs > self._sigma_upper * std_devs
            new_mask = lower | upper

            # UPDATE OUTPUT
            changed = bool(new_mask.any())
            output[new_mask] = np.nan
            iterations += 1

        # FILTER NaNs
        isnan = np.isnan(output)
        output[isnan] = centers[isnan]
        if self._masked_array: return cast(Output, ma.masked_array(output, mask=isnan))
        return cast(Output, output)

    @Decorators.running_time
    def _get_center(self, data: np.ndarray) -> np.ndarray:
        """
        To get the sliding median of 'data' given a kernel size.
        The kernel is cubic for now.

        Args:
            data (np.ndarray): the data to get the sliding median for.

        Returns:
            np.ndarray: the sliding median result.
        """

        pad = self._size // 2

        padded = np.pad(
            array=data,
            pad_width=pad,
            mode=cast(Literal['edge'], self._padding_mode),
            reflect_type=cast(Literal['even'], self._padding_reflect_type),
            constant_values=self._padding_constant_values,
        )

        if self._center_choice == 'mean': return _sliding_nanmean_3d(padded, self._size)
        return _sliding_nanmedian_3d(padded, self._size)
