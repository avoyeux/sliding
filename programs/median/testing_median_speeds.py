"""
Just to test partition vs median speeds as they were saying that they were using median.
"""
from __future__ import annotations

# IMPORTs alias
import numpy as np

# IMPORTs personal
from common import Decorators

# API public
__all__ = ["TestMedianSpeeds"]



class TestMedianSpeeds:
    """
    To test the speed difference between median and partition methods.
    """

    @Decorators.running_time
    def __init__(
            self,
            kernel_size: int = 3,
            iterations: int = 100,
        ) -> None:
        """
        Compares the processing speed of the np.median and np.partition methods for and odd
        sized cubic kernel.

        Args:
            kernel_size (int, optional): the size of the cubic kernel. The value must be odd for
                the median to be well defined. Defaults to 3.
            iterations (int, optional): the number of iterations to run the test. Defaults to 100.
        """

        self._kernel = np.random.rand(iterations, *[kernel_size] * 3).astype(np.float64)
        self._partition_index = self._kernel[0].size // 2
        self._iterations = iterations

        # RUN
        self._run()

    def _run(self) -> None:
        """
        Runs the median and partition speed tests and compares their results.
        """

        median_results = self._median()
        partition_results = self._partition()

        # COMPARISON
        all_equal = np.allclose(
            median_results,
            partition_results,
            rtol=1e-5,
            atol=1e-8,
            equal_nan=True,
        )
        print(f"Are median and partition results equal? {all_equal}")

    @Decorators.running_time
    def _median(self) -> np.ndarray[tuple[int, ...], np.dtype[np.floating]]:
        """
        Uses the np.median function to run a number of computations.

        Returns:
            np.ndarray[tuple[int, ...], np.dtype[np.floating]]: the array of computed median
                values.
        """

        results = np.empty((self._iterations,), dtype=self._kernel.dtype)
        for i in range(self._iterations):
            results[i] = np.median(self._kernel[i])
        return results

    @Decorators.running_time
    def _partition(self) -> np.ndarray[tuple[int, ...], np.dtype[np.floating]]:
        """
        Uses the np.partition function to run a number of computations.

        Returns:
            np.ndarray[tuple[int, ...], np.dtype[np.floating]]: the array of computed median
                values.
        """

        results = np.empty((self._iterations,), dtype=self._kernel.dtype)
        for i in range(self._iterations):
            part = np.partition(self._kernel[i].ravel(), self._partition_index)
            results[i] = part[self._partition_index]
        return results



if __name__ == "__main__": TestMedianSpeeds(kernel_size=3, iterations=1_000_000)
