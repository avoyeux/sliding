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
from programs.sigma_clipping import SlidingMean, SlidingMedian, FastStandardDeviation

# TYPE ANNOTATIONs
import queue

# API public
__all__ = ['TestMeanMedian']



class TestMeanMedian:
    """
    To compare the new mean and median implementations with the old generic filter based approach.
    """

    @pytest.fixture(scope="class")
    def filepaths(self) -> list[str]:
        """
        Gives the FITs filepaths to the test FITS files.
        """

        filepaths = TestUtils.get_filepaths()
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
            # kernel = np.ones(kernel_size, dtype=data.dtype)
            # new_mean = SlidingMean(
            #     data=data,
            #     kernel=kernel,
            #     borders='reflect',
            #     threads=1,
            # ).mean

            std_instance = FastStandardDeviation(
                data=data,
                kernel=kernel_size,
                borders='reflect',
            )
            new_mean = std_instance.mean

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
            new_median = SlidingMedian(
                data=data,
                kernel=kernel,
                borders='reflect',
                threads=1,
            ).median

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
            new_median = SlidingMedian(
                data=data,
                kernel=kernel_size,
                borders='reflect',
                threads=1,
            ).median

            # Comparison
            comparison_log = TestUtils.compare(
                actual=old_median,
                desired=new_median,
                filepath=filepath,
            )
            result_queue.put(comparison_log)
