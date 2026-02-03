"""
Code to perform n-dimensional sigma clipping on numpy arrays (with/without NaNs) with a kernel
(with/without weights).
"""
from __future__ import annotations

# IMPORTs alias
import numpy as np

# IMPORTs sub
from numpy import ma
from numba import set_num_threads

# IMPORTs local
from sliding.convolution import BorderType
from sliding.mode import SlidingMedian
from sliding.standard_deviation import SlidingStandardDeviation

# TYPE ANNOTATIONs
from typing import cast, Literal, TypeAlias
KernelType: TypeAlias = int | tuple[int, ...] | np.ndarray[tuple[int, ...], np.dtype[np.floating]]

# API public
__all__ = ["SlidingSigmaClipping"]

# SENTINEL for default sigma lower/upper
class _UseSigma: pass
_USE_SIGMA = _UseSigma()

# todo test the code on 1D data.



class SlidingSigmaClipping:
    """
    For a n-dimensional sliding sigma clip of an input array (with/without NaNs) with a given
    kernel (with/without weights).
    Use np.float32 or np.float64 arrays as input data (np.float64 recommended).
    The kernel must have odd dimensions.

    Raises:
        ValueError: if the kernel size is even.
    """

    def __init__(
            self,
            data: np.ndarray,
            kernel: KernelType = 3,
            center_choice: Literal['median', 'mean'] = 'median',
            sigma: float = 3.,
            sigma_lower: float | None | _UseSigma = _USE_SIGMA,
            sigma_upper: float | None | _UseSigma = _USE_SIGMA,
            max_iters: int | None = 5,
            borders: BorderType = 'reflect',
            threads: int | None = 1,
            masked_array: bool = True,
        ) -> None:
        """
        Runs the sliding sigma clipping where the flagged pixels are swapped with the center value
        (mean or median) for that pixel. The kernel needs to be of the same dimensions than the
        input data and have positive odd dimensions.
        The result is accessed through the 'results' property and will be a numpy.ma.MaskedArray if
        'masked_array' is set to True, else a numpy.array. The sigma clipping is done iteratively
        'max_iters' number of times or till there are no more pixels flagged.
        ! Uses a lot of memory, ~kernel.size * input_data.nbytes.

        Args:
            data (np.ndarray): the data to sigma clip.
            kernel (KernelType, optional): the kernel information used for computing the modes
                and standard deviations. Can be an int (square kernel), a tuple of ints (defining
                the shape of the kernel) or a numpy ndarray (defining the full kernel with
                weights). Defaults to 3.
            center_choice (Literal['median', 'mean'], optional): the function to use for computing
                the mode for each pixel. Defaults to 'median'.
            sigma (float): the number of standard deviations to use for both the lower and upper
                clipping limit. Overridden by 'sigma_lower' and/or 'sigma_upper'. 
            sigma_lower (float | None | _UseSigma, optional): the number of standard deviations to
                use for the lower clipping limit. It will be set to 'sigma' if _USE_SIGMA. When set
                to None, no lower clipping is done. Defaults to _USE_SIGMA.
            sigma_upper (float | None | _UseSigma, optional): the number of standard deviations to
                use for the upper clipping limit. It will be set to 'sigma' if _USE_SIGMA. When set
                to None, no upper clipping is done. Defaults to _USE_SIGMA.
            max_iters (int | None, optional): the maximum number of iterations to perform.
                If None, iterate until convergence. Defaults to 5.
            borders (BorderType, optional): the type of borders to use. These are the type of
                borders used by OpenCV (not all OpenCV borders are implemented as some don't have
                the equivalent in np.pad or scipy.ndimage). If None, uses adaptative borders, i.e.
                no padding and hence smaller kernels at the borders. Defaults to 'reflect'.
            threads (int | None, optional): the number of threads to use for numba parallelization.
                when setting 'center_choice' to 'median'. If None, uses the default number of
                threads. Not used for the standard deviation. Defaults to 1.
            masked_array (bool, optional): whether to return a MaskedArray (True) or a normal
                ndarray (False). Defaults to True.

        Raises:
            ValueError: if both 'sigma_lower' and 'sigma_upper' are None.
        """

        self._data = data
        self._sigma = sigma
        self._kernel = kernel
        self._borders = borders
        self._threads = threads
        self._masked_array = masked_array
        self._center_choice = center_choice
        self._sigma_lower = sigma if sigma_lower is _USE_SIGMA else sigma_lower
        self._sigma_upper = sigma if sigma_upper is _USE_SIGMA else sigma_upper
        self._max_iters = cast(int, max_iters if max_iters is not None else np.inf)

        # CHECK
        if (self._sigma_upper is None) and (self._sigma_lower is None):
            raise ValueError("At least one 'sigma_upper' or 'sigma_lower' must be not None.")

        # RUN
        if self._threads is not None: set_num_threads(self._threads)
        self._sigma_clipped = self._run()

    @property
    def results(self) -> np.ndarray | ma.MaskedArray:
        """
        The sigma clipped input where the pixels that where flagged are set to the mode value
        (median or mean) gotten for that pixel.
        If 'masked_array' is True, the output is a MaskedArray where the masked pixels are the
        ones that were flagged during the sigma clipping process. Else, it's a normal ndarray.

        Returns:
            np.ndarray | ma.MaskedArray: the sigma clipped data.
        """
        return self._sigma_clipped

    def _run(self) -> np.ndarray | ma.MaskedArray:
        """
        Runs the sigma clipping code where flagged pixels are set to the mode value gotten for that
        pixel (and after all iterations are done).
        The sigma clipping is done iteratively 'max_iters' number of times or till there are no
        more pixels that are flagged.
        If 'masked_array' is True, the output is a MaskedArray where the masked pixels are the
        ones that were flagged during the sigma clipping process.

        Returns:
            np.ndarray | ma.MaskedArray: the sigma clipped data.
        """

        output = self._data.copy()

        # TYPE CHECKER complains
        mode = np.empty(0, dtype=self._data.dtype)
        self._borders = cast(BorderType, self._borders)
        self._sigma_lower = cast(float | None, self._sigma_lower)
        self._sigma_upper = cast(float | None, self._sigma_upper)

        # COUNTs
        changed: bool = True
        iterations = 0

        # BUFFER mask
        new_mask = np.zeros(self._data.shape, dtype=np.bool_)

        # LOOP
        while (changed is True) and (iterations < self._max_iters):
            # BUFFER reset
            new_mask.fill(False)

            # CENTERs and STDDEVs
            instance_std = SlidingStandardDeviation(
                data=output,
                kernel=self._kernel,
                borders=self._borders,
            )
            std_devs = instance_std.standard_deviation

            # MODE
            if self._center_choice == 'mean':
                mode = instance_std.mean
            elif self._center_choice == 'median':
                mode = SlidingMedian(
                    data=output,
                    kernel=self._kernel,
                    borders=self._borders,#type: ignore
                    threads=self._threads,
                ).median
            else:
                raise ValueError(f"Unknown center choice: {self._center_choice}")

            diffs = output - mode

            # MASK update
            if self._sigma_lower is not None:
                np.logical_or(new_mask, diffs < - self._sigma_lower * std_devs, out=new_mask)
            if self._sigma_upper is not None:
                np.logical_or(new_mask, diffs > self._sigma_upper * std_devs, out=new_mask)

            # UPDATE OUTPUT
            output[new_mask] = np.nan
            iterations += 1
            changed = bool(new_mask.any())

        # FILTER NaNs
        isnan = np.isnan(output)
        output[isnan] = mode[isnan]
        if self._masked_array: return ma.masked_array(output, mask=isnan)
        return output
