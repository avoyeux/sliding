"""
To compare the efficiency and results of the different implementations of the moving sample
standard deviation computations.
"""
from __future__ import annotations

# IMPORTs alias
import numpy as np

# IMPORTs local
from programs.sigma_clipping.old_sigma_clipping import sigma_clip as old_sigma_clip
from programs.sigma_clipping.sigma_clipping_test import FastSigmaClipping

# IMPORTs personal
from common import Decorators


class CompareClipping:
    # todo add docstring

    @Decorators.running_time
    def __init__(
            self,
            data: np.ndarray,
            kernel_size: int,
            sigma: float = 3.,
            sigma_lower: float | None = None,
            sigma_upper: float | None = None,
            max_iters: int | None = 5,
            tolerance: float = 1e-5,
            abs_tol: float = 1e-8,
        ) -> None:

        self._data = data
        self._kernel_size = kernel_size
        self._tolerance = tolerance
        self._abs_tol = abs_tol
        self._sigma = sigma
        self._sigma_lower = sigma_lower
        self._sigma_upper = sigma_upper
        self._max_iters = max_iters

        # RUN
        self._run()

    def _run(self) -> None:
        # todo add docstring

        new_result = FastSigmaClipping(
            data=self._data,
            size=self._kernel_size,
            sigma=self._sigma,
            center_choice='median',
            sigma_lower=self._sigma_lower,
            sigma_upper=self._sigma_upper,
            max_iters=self._max_iters,
            masked_array=False,
        ).results

        new_result2 = FastSigmaClipping(
            data=self._data,
            size=self._kernel_size,
            center_choice='mean',
            sigma=self._sigma,
            sigma_lower=self._sigma_lower,
            sigma_upper=self._sigma_upper,
            max_iters=self._max_iters,
            masked_array=False,
        ).results

        old_result = old_sigma_clip(
            data=self._data,
            size=self._kernel_size,
            sigma=self._sigma,
            sigma_lower=self._sigma_lower,
            sigma_upper=self._sigma_upper,
            max_iters=self._max_iters,
            center_func='median',
            masked=False,
        )

        old_result2 = old_sigma_clip(
            data=self._data,
            size=self._kernel_size,
            sigma=self._sigma,
            sigma_lower=self._sigma_lower,
            sigma_upper=self._sigma_upper,
            max_iters=self._max_iters,
            center_func='mean',
            masked=False,
        )

        # COMPARE
        self._print_comparison(
            name1="Old Sigma Clipping (median)",
            name2="New Sigma Clipping (median)",
            value1=old_result,
            value2=new_result,
        )
        self._print_comparison(
            name1="Old Sigma Clipping (mean)",
            name2="New Sigma Clipping (mean)",
            value1=old_result2,
            value2=new_result2,
        )

    def _print_comparison(self, name1: str, name2: str, value1: np.ndarray, value2: np.ndarray) -> None:
        """
        Prints a comparison between two arrays.

        Args:
            value1 (np.ndarray): first array.
            value2 (np.ndarray): second array.
        """

        all_equal = np.allclose(
            a=value1,
            b=value2,
            rtol=self._tolerance,
            atol=self._abs_tol,
            equal_nan=True,
        )

        if all_equal:
            print(
                f">>> SUCCESS: {name1} and {name2} are equal! "
                f"(min, max) are ({np.nanmin(value1):.5f}, {np.nanmax(value1):.5f})"
                f"and ({np.nanmin(value2):.5f}, {np.nanmax(value2):.5f})"
            )
        else:
            elements_equal = np.isclose(
                a=value1,
                b=value2,
                rtol=self._tolerance,
                atol=self._abs_tol,
                equal_nan=True,
            )
            n_elements_equal = np.sum(elements_equal)
            n_elements_total = elements_equal.size
            percentage_equal = n_elements_equal / n_elements_total * 100.0
            print(
                f">>> FAILURE: {name1} and {name2} are NOT equal! "
                f"Only {n_elements_equal} / {n_elements_total} "
                f"({percentage_equal:.2f} %) elements are equal."
            )



if __name__ == "__main__":

    data = np.random.rand(36, 1024, 128).astype(np.float64)
    data[5:10, 100:200, 50: 80] = 10.
    data[15:20, 500:600, 90: 120] = 3.
    data[2:7, :, 10:90] = 20.

    # RUN comparison
    CompareClipping(
        data=data,
        kernel_size=3,
        sigma=2,
        max_iters=1,
        tolerance=1e-5,
        abs_tol=1e-8,
    )
