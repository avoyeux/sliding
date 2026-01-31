"""
To test if the sliding mean works as intended with weights.
Still don't know how to properly test the sliding median, sliding standard deviation and the
sliding sigma clipping with weights.
Furthermore, I am still not testing for the sliding mean with weights when there are NaNs in the
data.
"""
from __future__ import annotations

# IMPORTs
import pytest

# IMPORTs sub
from scipy.ndimage import convolve

# IMPORTs alias
import numpy as np

# IMPORTs local
from tests.utils import TestUtils
from sliding import SlidingMean, SlidingStandardDeviation

# TYPE ANNOTATIONs
import queue



TestUtils.ADD_NANS = False
class TestWeightsNoNaNs:
    """
    To test if the sliding mean works as intended with weights but no NaNs in the data.
    """

    @pytest.fixture(scope="class")
    def filepaths(self) -> list[str]:
        """
        Gives the FITs filepaths to the test FITS files.
        """

        filepaths = TestUtils.get_filepaths()
        print(f"Found {len(filepaths)} FITs files for testing weights with no NaNs.")
        return filepaths

    def test_mean_sliding_weights_no_nans(
            self,
            filepaths: list[str],
        ) -> None:
        """
        Running tests on the new numba mean implementation with weights and no NaNs.

        Args:
            filepaths (list[str]): the FITS filepaths to run the tests for.
        """

        TestUtils.multiprocess(
            filepaths=filepaths,
            target=self._run_mean_sliding_weights_no_nans,
        )

    def test_mean_std_weights_no_nans(
            self,
            filepaths: list[str],
        ) -> None:
        """
        Running tests on the new mean from the standard deviation implementation with weights and
        no NaNs.

        Args:
            filepaths (list[str]): the FITS filepaths to run the tests for.
        """

        TestUtils.multiprocess(
            filepaths=filepaths,
            target=self._run_mean_std_weights_no_nans,
        )

    @staticmethod
    def _run_mean_sliding_weights_no_nans(
            input_queue: queue.Queue[str | None],
            result_queue: queue.Queue[dict],
        ) -> None:
        """
        Old and new mean implementation test for the mean with weights and no NaNs.

        Args:
            input_queue (queue.Queue[str | None]): the input queue with the filepaths.
            result_queue (queue.Queue[dict]): the result queue to put the results in.
        """

        kernel_size = 5

        while True:
            filepath = input_queue.get()
            if filepath is None:
                break

            data = TestUtils.open_file(filepath)
            if isinstance(data, dict): result_queue.put(data); continue

            kernel = np.ones((kernel_size,) * data.ndim, dtype=np.float64)
            middle = tuple((size // 2) for size in kernel.shape)
            kernel[middle] = 0.

            # NEW mean with weights
            new_result = SlidingMean(
                data=data,
                kernel=kernel,
                borders='reflect',
            ).mean

            # OLD mean with weights
            kernel /= kernel.sum()
            output = np.empty(data.shape, dtype=data.dtype)
            convolve(
                data,
                kernel,
                output=output,
                mode='reflect',
            )

            # CHECK all values
            comparison_log = TestUtils.compare(
                actual=new_result,
                desired=output,
                filepath=filepath,
            )
            result_queue.put(comparison_log)

    @staticmethod
    def _run_mean_std_weights_no_nans(
            input_queue: queue.Queue[str | None],
            result_queue: queue.Queue[dict],
        ) -> None:
        """
        Old and new mean implementation test for the mean with weights and no NaNs (using the 
        standard deviation class).

        Args:
            input_queue (queue.Queue[str | None]): the input queue with the filepaths.
            result_queue (queue.Queue[dict]): the result queue to put the results in.
        """

        kernel_size = 5

        while True:
            filepath = input_queue.get()
            if filepath is None:
                break

            data = TestUtils.open_file(filepath)
            if isinstance(data, dict): result_queue.put(data); continue

            kernel = np.ones((kernel_size,) * data.ndim, dtype=np.float64)
            middle = tuple((size // 2) for size in kernel.shape)
            kernel[middle] = 0.

            # NEW mean with weights
            new_result = SlidingStandardDeviation(
                data=data,
                kernel=kernel,
                borders='reflect',
            ).mean

            # OLD mean with weights
            kernel /= kernel.sum()
            output = np.empty(data.shape, dtype=data.dtype)
            convolve(
                data,
                kernel,
                output=output,
                mode='reflect',
            )

            # CHECK all values
            comparison_log = TestUtils.compare(
                actual=new_result,
                desired=output,
                filepath=filepath,
            )
            result_queue.put(comparison_log)
