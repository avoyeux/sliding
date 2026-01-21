"""
To test if the sliding and mean implementations between the new code and generic filter do give
the same results.
"""
from __future__ import annotations

# IMPORTs alias
import numpy as np

# IMPORTs third-party
import pytest

# IMPORTs local
from programs.tests.utils_tests import TestUtils
from programs.standard_deviation import Convolution
from programs.sigma_clipping.numba_functions import (
    tuple_sliding_nanmedian_3d, sliding_weighted_median_3d,
    tuple_sliding_nanmedian_nd, sliding_weighted_median_nd,
)

# TYPE ANNOTATIONs
import queue



class TestMeanMedian:
    """
    To compare the new mean and median implementations with the old generic filter based approach.
    """

    @pytest.fixture(scope="class")
    def filepaths(self) -> list[str]:
        """
        Gives the FITs filepaths to the test FITS files.
        """

        filepaths = TestUtils.get_filepaths()[120:]
        print(f"Found {len(filepaths)} FITs files for testing mean and median.")
        return filepaths

    def test_mean(
            self,
            filepaths: list[str],
        ) -> None:
        """
        Running tests on the new and old mean implementations.

        Args:
            filepaths (list[str]): the FITS filepaths to run the tests for.
        """

        TestUtils.multiprocess(
            filepaths=filepaths,
            target=self._run_mean_test,
        )

    def test_median_array(
            self,
            filepaths: list[str],
        ) -> None:
        """
        Running tests on the new and old median implementations when the kernel is a numpy array
        (used when needing to add weights).

        Args:
            filepaths (list[str]): the FITS filepaths to run the tests for.
        """

        TestUtils.multiprocess(
            filepaths=filepaths,
            target=self._run_median_array_test,
        )

    def test_median_tuple(
            self,
            filepaths: list[str],
        ) -> None:
        """
        Running test on the new and old median implementations when the kernel is an int or a
        tuple of ints.

        Args:
            filepaths (list[str]): the FITS filepaths to run the tests for.
        """

        TestUtils.multiprocess(
            filepaths=filepaths,
            target=self._run_median_tuple_test,
        )

    @staticmethod
    def _run_mean_test(
            input_queue: queue.Queue[str | None],
            result_queue: queue.Queue[dict],
        ) -> None:
        """
        Old and new mean implementation test (used as the target for the multiprocessing).

        Args:
            input_queue (queue.Queue[str | None]): the input queue with the filepaths.
            result_queue (queue.Queue[dict]): the result queue to put the results in.
        """

        while True:
            filepath = input_queue.get()
            if filepath is None: break

            data = TestUtils.open_file(filepath)
            if isinstance(data, dict): result_queue.put(data); continue

            # OLD mean
            kernel_size = (3,) * data.ndim
            old_mean = TestUtils.old_implementation(
                function='mean',
                data=data,
                kernel_size=kernel_size,
            )

            # NEW mean
            kernel = np.ones(kernel_size, dtype=data.dtype)
            new_mean = TestMeanMedian._new_mean_implementation(
                data=data,
                kernel=kernel,
                borders='reflect',
            )

            # Comparison
            comparison_log = TestUtils.compare(
                actual=old_mean,
                desired=new_mean,
                filepath=filepath,
            )
            result_queue.put(comparison_log)

    @staticmethod
    def _run_median_array_test(
            input_queue: queue.Queue[str | None],
            result_queue: queue.Queue[dict],
        ) -> None:
        """
        Old and new mean implementation test when the kernel has weights (used as the target for
        the multiprocessing).

        Args:
            input_queue (queue.Queue[str | None]): the input queue with the filepaths.
            result_queue (queue.Queue[dict]): the result queue to put the results in.
        """

        while True:
            filepath = input_queue.get()
            if filepath is None: break

            data = TestUtils.open_file(filepath)
            if isinstance(data, dict): result_queue.put(data); continue

            # OLD median
            kernel_size = (3,) * data.ndim
            old_median = TestUtils.old_implementation(
                function='median',
                data=data,
                kernel_size=kernel_size,
            )

            # NEW median
            kernel = np.ones(kernel_size, dtype=data.dtype)
            new_median = TestMeanMedian._new_median_array_implementation(
                data=data,
                kernel=kernel,
                borders='reflect',
            )

            # Comparison
            comparison_log = TestUtils.compare(
                actual=old_median,
                desired=new_median,
                filepath=filepath,
            )
            result_queue.put(comparison_log)

    @staticmethod
    def _run_median_tuple_test(
            input_queue: queue.Queue[str | None],
            result_queue: queue.Queue[dict],
        ) -> None:
        """
        Old and new mean implementation test when the kernel doesn't need weights (used as the
        target for the multiprocessing).

        Args:
            input_queue (queue.Queue[str | None]): the input queue with the filepaths.
            result_queue (queue.Queue[dict]): the result queue to put the results in.
        """

        while True:
            filepath = input_queue.get()
            if filepath is None: break

            data = TestUtils.open_file(filepath)
            if isinstance(data, dict): result_queue.put(data); continue

            # OLD median
            kernel_size = (3,) * data.ndim
            old_median = TestUtils.old_implementation(
                function='median',
                data=data,
                kernel_size=kernel_size,
            )

            # NEW median
            kernel = tuple(3 for _ in range(data.ndim))
            new_median = TestMeanMedian._new_median_tuple_implementation(
                data=data,
                kernel=kernel,
                borders='reflect',
            )

            # Comparison
            comparison_log = TestUtils.compare(
                actual=old_median,
                desired=new_median,
                filepath=filepath,
            )
            result_queue.put(comparison_log)

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
            mean = np.where(count > 0, sum_values / count, 0.0)
        return mean

    @staticmethod
    def _new_median_array_implementation(
            data: np.ndarray,
            kernel: np.ndarray,
            borders: str | None,
        ) -> np.ndarray:
        """
        The new median implementation when the kernel has weights.

        Args:
            data (np.ndarray): the data to get the sliding median for.
            kernel (np.ndarray): the weighted kernel for the sliding median.
            borders (str | None): the border type.

        Returns:
            np.ndarray: the sliding median result.
        """

        pad = tuple((k // 2, k // 2) for k in kernel.shape)
        padded = TestUtils.add_padding(borders, data, pad)

        if kernel.ndim == 3: return sliding_weighted_median_3d(padded, kernel)
        return sliding_weighted_median_nd(padded, kernel)

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
