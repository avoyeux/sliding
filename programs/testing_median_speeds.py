"""
Just to test partition vs median speeds as they were saying that they were using median.
"""

# IMPORTs alias
import numpy as np

# IMPORTs personal
from common import Decorators



class TestMedianSpeeds:
    # todo add docstring

    @Decorators.running_time
    def __init__(
        self,
        # data: np.ndarray[tuple[int, ...], np.dtype[np.floating]],
        kernel_size: int = 3,
        iterations: int = 100,
    ) -> None:
        
        # todo need to decide how to do so when using a kernel

        # self._data = data
        # self._kernel = (
        #     np.ones((kernel_size,) * data.ndim, dtype=np.float64) / (kernel_size ** data.ndim)
        # )
        self._kernel = np.random.rand(iterations, *[kernel_size] * 3).astype(np.float64)
        self._iterations = iterations

    def _run(self) -> None:
        # todo add docstring

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
        # todo add docstring

        results = np.empty((self._iterations,), dtype=self._kernel.dtype)
        for i in range(self._iterations):
            results[i] = np.median(self._kernel[i])
        return results

    @Decorators.running_time
    def _partition(self) -> np.ndarray[tuple[int, ...], np.dtype[np.floating]]:
        # todo add docstring

        results = np.empty((self._iterations,), dtype=self._kernel.dtype)
        for i in range(self._iterations):
            part = np.partition(self._kernel[i].ravel(), self._kernel[i].size // 2)
            results[i] = part[part.size // 2]
        return results


if __name__ == "__main__":
    
    TestMedianSpeeds(
        # data=np.random.rand(1024, 1024).astype(np.float64),
        kernel_size=5,
        iterations=1_000_000,
    )._run()
