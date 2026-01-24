"""
Mixed test with a mix of different implementations mainly done so to try and catch were the small
differences between the final values after the sigma clipping comes from.
"""
from __future__ import annotations

# IMPORTs third-party
import pytest

# IMPORTs alias
import numpy as np

# IMPORTs local
from programs.tests.utils_tests import TestUtils
from programs.sigma_clipping import FastStandardDeviation, Convolution, sigma_clip
from programs.sigma_clipping.sliding_mode.numba_functions import (
    tuple_sliding_nanmedian_3d, tuple_sliding_nanmedian_nd,
)

# TYPE ANNOTATIONs
import queue
from typing import Literal, cast



class MixedSigmaClip:
    """
    Contains the old generic filter code but the actual generic filter calls are replaced with
    the corresponding new implementations.
    """

    def __init__(
            self,
            data: np.ndarray,
            method: Literal['mean', 'median'],
            size: tuple[int, ...],
            sigma: float = 3.,
            max_iters: int | None = 5,
        ) -> None:
        # todo add docstring
        
        self._data = data
        self._size = size
        self._sigma = sigma
        self._method = method
        self._max_iters = max_iters

    def sigma_clip(self) -> np.ndarray:
        """
        Returns the sigma clipped array.

        Returns:
            np.ndarray: the sigma clipped data.
        """

        output = self._data.copy()

        max_iters = cast(int, self._max_iters or np.inf)
        n_changed = 1
        iteration = 0
        # PLACEHOLDER
        center = np.empty(0, dtype=self._data.dtype)
        while n_changed != 0 and (iteration < max_iters):
            iteration += 1
            center = self._get_center(output)
            stddev = FastStandardDeviation(
                data=output,
                kernel=self._size,
                borders='reflect',
                with_NaNs=True,
            ).sdev
            diff = output - center
            new_mask = (diff > self._sigma * stddev) | (diff < - self._sigma * stddev)
            output[new_mask] = np.nan
            n_changed = np.count_nonzero(new_mask)

        nan = np.isnan(output)
        output[nan] = center[nan]
        return output

    def _get_center(self, data: np.ndarray) -> np.ndarray:
        """
        To get the sliding center value using the new implementation.
        """

        if self._method == 'mean':
            kernel = np.ones(self._size, dtype=data.dtype)
            centers = self._new_mean_implementation(
                data=data,
                kernel=kernel,
                borders='reflect',
            )
        elif self._method == 'median':
            centers = self._new_median_tuple_implementation(
                data=data,
                kernel=self._size,
                borders='reflect',
            )
        else:
            raise ValueError(f"Unknown method '{self._method}' for center calculation.")
        return centers

    @staticmethod
    def _new_mean_implementation(
            data: np.ndarray,
            kernel: np.ndarray,
            borders: str | None,
        ) -> np.ndarray:
        """
        The new mean implementation used in the new sigma clipping method.

        Args:
            data (np.ndarray): the data to get the sliding mean for.
            kernel (np.ndarray): the kernel for the sliding mean.
            borders (str | None): the border type.

        Returns:
            np.ndarray: the sliding mean result.
        """

        # NaN handling
        valid_mask = ~np.isnan(data)
        data_filled = np.where(valid_mask, data, 0.)

        # SUM n MEAN
        sum_values = Convolution(
            data=data_filled,
            kernel=kernel,
            borders=borders,#type:ignore
            threads=1,
        ).result
        count = Convolution(
            data=valid_mask.astype(data.dtype),
            kernel=np.ones(kernel.shape, dtype=data.dtype),
            borders=borders,#type:ignore
            threads=1,
        ).result
        with np.errstate(divide='ignore', invalid='ignore'):
            mean = np.where(count > 0, sum_values / count, 0.)
        return mean

    @staticmethod
    def _new_median_tuple_implementation(
            data: np.ndarray,
            kernel: tuple[int, ...],
            borders: str | None,
        ) -> np.ndarray:
        """
        The new median implementation when the kernel doesn't have weights.

        Args:
            data (np.ndarray): the data to get the sliding median for.
            kernel (tuple[int, ...]): the kernel for the sliding median.
            borders (str | None): the border type.

        Returns:
            np.ndarray: the sliding median result.
        """

        # MEDIAN
        pad = tuple((k // 2, k // 2) for k in kernel)
        padded = TestUtils.add_padding(borders, data, pad)

        # MEDIAN choice
        if len(kernel) == 3: return tuple_sliding_nanmedian_3d(padded, kernel)#type:ignore
        return tuple_sliding_nanmedian_nd(padded, kernel)


class TestMixedCase:
    """
    Test of the mixed sigma clipping implementation with the old implementation.
    """

    @pytest.fixture(scope='class')
    def filepaths(self) -> list[str]:
        """
        Gives the FITs filepaths to the test FITS files.
        """

        filepaths = TestUtils.get_filepaths()
        print(f"Found {len(filepaths)} FITs files for testing mixed sigma clipping.")
        return filepaths

    def test_mixed_mean(
            self,
            filepaths: list[str],
        ) -> None:
        """
        Running tests on the mixed sigma clipping implementation with mean center.

        Args:
            filepaths (list[str]): the FITs filepaths to run the tests for.
        """

        TestUtils.multiprocess(
            filepaths=filepaths,
            target=self._run_mixed_mean_test,
        )

    # def test_mixed_median(
    #         self,
    #         filepaths: list[str],
    #     ) -> None:
    #     """
    #     Running tests on the mixed sigma clipping implementation with median center.

    #     Args:
    #         filepaths (list[str]): the FITs filepaths to run the tests for.
    #     """

    #     TestUtils.multiprocess(
    #         filepaths=filepaths,
    #         target=self._run_mixed_median_test,
    #     )

    @staticmethod
    def _run_mixed_mean_test(
            input_queue: queue.Queue[str | None],
            result_queue: queue.Queue[dict],
        ) -> None:
        """
        Mixed sigma clipping implementation test with mean center.

        Args:
            input_queue (queue.Queue[str | None]): the input queue with the filepaths.
            result_queue (queue.Queue[dict]): the result queue to put the results in.
        """

        while True:
            filepath = input_queue.get()
            if filepath is None: break

            data = TestUtils.open_file(filepath)
            if isinstance(data, dict): result_queue.put(data); continue

            kernel = (3,) * data.ndim
            mixed_result = MixedSigmaClip(
                data=data,
                method='mean',
                size=kernel,
                sigma=1.,
                max_iters=1,
            ).sigma_clip()

            old_result = sigma_clip(
                data=data,
                size=kernel,
                sigma=1.,
                max_iters=1,
                center_func='mean',
                masked=False,
            )

            comparison_log = TestUtils.compare(
                desired=mixed_result,
                actual=old_result,
                filepath=filepath,
            )
            result_queue.put(comparison_log)

    @staticmethod
    def _run_mixed_median_test(
            input_queue: queue.Queue[str | None],
            result_queue: queue.Queue[dict],
        ) -> None:
        """
        Mixed sigma clipping implementation test with median center.

        Args:
            input_queue (queue.Queue[str | None]): the input queue with the filepaths.
            result_queue (queue.Queue[dict]): the result queue to put the results in.
        """

        while True:
            filepath = input_queue.get()
            if filepath is None: break

            data = TestUtils.open_file(filepath)
            if isinstance(data, dict): result_queue.put(data); continue

            kernel = (3,) * data.ndim
            mixed_result = MixedSigmaClip(
                data=data,
                method='median',
                size=kernel,
                sigma=2.,
                max_iters=2,
            ).sigma_clip()

            old_result = sigma_clip(
                data=data,
                size=kernel,
                sigma=2.,
                max_iters=2,
                center_func='median',
                masked=False,
            )

            comparison_log = TestUtils.compare(
                desired=mixed_result,
                actual=old_result,
                filepath=filepath,
            )
            result_queue.put(comparison_log)
