"""
Code to test if the border cases are handled the same way between different implementations.
"""
from __future__ import annotations

# IMPORTs third-party
import pytest

# IMPORTs local
from programs.tests.utils_tests import TestUtils
from programs.standard_deviation import FastStandardDeviation

# TYPE ANNOTATIONs
import queue



class TestStandardDeviation:
    """
    To compare the values gotten from the old and new standard deviation implementations.
    """

    @pytest.fixture(scope="class")
    def filepaths(self) -> list[str]:
        """
        Gives the FITs filepaths to the test FITS files.
        """

        filepaths = TestUtils.get_filepaths()
        print(f"Found {len(filepaths)} FITs files for testing borders.")
        return filepaths

    def test_standard_deviation(
            self,
            filepaths: list[str],
        ) -> None:
        """
        Running tests on the new and old standard deviation implementations.

        Args:
            filepaths (list[str]): the FITS filepaths to run the tests for.
        """

        TestUtils.multiprocess(
            filepaths=filepaths,
            target=self._run_process,
        )

    @staticmethod
    def _run_process(
            input_queue: queue.Queue[str | None],
            result_queue: queue.Queue[dict],
        ) -> None:
        """
        Old and new mean implementation test for the standard deviation.

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

            # OLD std
            old_standard_deviation = TestUtils.old_implementation(
                function='std',
                data=data,
                kernel_size=kernel,
            )

            # NEW std
            new_standard_deviation = FastStandardDeviation(
                data=data,
                kernel=kernel,
                borders='reflect',
                with_NaNs=True,
                threads=1,
            ).sdev

            # print(f"old and new dtypes are {old_standard_deviation.dtype} and {new_standard_deviation.dtype}", flush=True)

            # COMPARISON
            comparison_log = TestUtils.compare(
                actual=new_standard_deviation,
                desired=old_standard_deviation,
                filepath=filepath,
            )
            result_queue.put(comparison_log)
