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

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

def sliding(arr: np.ndarray, kernel_shape: tuple[int, ...]):
    # todo add docstring

    axis = tuple(range(arr.ndim))
    assert len(kernel_shape) == len(axis), "kernel_shape match number of axes"
    
    pad = tuple((k // 2, k // 2) for k in kernel_shape)
    arr = TestUtils.add_padding(
        border='reflect',
        data=arr,
        pad=pad,
    )

    # VIEW
    windows = sliding_window_view(
        x=arr,
        window_shape=kernel_shape,
        axis=axis,#type: ignore
    )

    kernel_axes = tuple(range(-len(axis), 0))

    # MEAN
    mean = windows.mean(axis=kernel_axes)

    # STD stable
    diff = windows - np.expand_dims(mean, axis=kernel_axes)
    M2 = np.sum(diff**2, axis=kernel_axes)
    std = np.sqrt(M2 / np.prod(kernel_shape))
    return mean, std


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

            # # OLD std
            # old_standard_deviation = TestUtils.old_implementation(
            #     function='std',
            #     data=data,
            #     kernel_size=kernel,
            # )

            # NEW std
            new_standard_deviation = FastStandardDeviation(
                data=data,
                kernel=kernel,
                borders='reflect',
                with_NaNs=True,
                threads=1,
            ).sdev
            # _, new_standard_deviation = sliding_welford(
            #     arr=data,
            #     kernel_shape=kernel,
            #     circular=False,
            #     axis=None,
            # )

            # print(f"old and new dtypes are {old_standard_deviation.dtype} and {new_standard_deviation.dtype}", flush=True)

            # # COMPARISON
            # comparison_log = TestUtils.compare(
            #     actual=new_standard_deviation,
            #     desired=old_standard_deviation,
            #     filepath=filepath,
            # )
            # result_queue.put(comparison_log)
