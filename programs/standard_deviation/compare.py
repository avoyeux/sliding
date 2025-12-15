"""
Code to compare the run times and results of different moving sample standard deviation
implementations.
"""
from __future__ import annotations

# IMPORTs alias
import numpy as np

# IMPORTs local
from programs.standard_deviation.convolve_3d import Convolution3D
from programs.standard_deviation.frederic_convolution import StandardDeviation
from programs.standard_deviation.sospice_generic_filter import GenericFilter

# IMPORTs personal
from common import Decorators

# TYPE ANNOTATIONs
from typing import Any

# API public
__all__ = ["CompareSTDs"]



class CompareSTDs:
    """
    To compare the different implementations of the moving sample standard deviation computations.
    """

    @Decorators.running_time
    def __init__(
            self,
            data: np.ndarray[tuple[int, ...], np.dtype[Any]],
            kernel_size: int = 5,
            tolerance: float = 1e-5,
            abs_tol: float = 1e-8,
        ) -> None:
        """
        Compares the different implementations of the moving sample standard deviation
        computations. The results are printed.

        Args:
            data (np.ndarray[tuple[int, ...], np.dtype[Any]]): the data for which to compute the
                moving sample standard deviation.
            kernel_size (int, optional): the size of the moving window. Defaults to 5.
            tolerance (float, optional): the relative tolerance for comparison. Defaults to 1e-5.
            abs_tol (float, optional): the absolute tolerance for comparison. Defaults to 1e-8.
        """

        self._data = data
        self._kernel_size = kernel_size
        self._tolerance = tolerance
        self._abs_tol = abs_tol

        # RUN
        self._compare()

    def _compare(self) -> None:
        """
        Compares the different implementations of the moving sample standard deviation
        computations.
        """

        print(f"The kernel size is {self._kernel_size}.")

        # STD computations
        legacy = GenericFilter(
            data=self._data,
            kernel_size=self._kernel_size,
            with_nans=False,
        )
        legacy_nan = GenericFilter(
            data=self._data,
            kernel_size=self._kernel_size,
            with_nans=True,
        )
        new = StandardDeviation(
            data=self._data,
            kernel=self._kernel_size,
        )
        new_nan = StandardDeviation(
            data=self._data,
            kernel=self._kernel_size,
            with_NaNs=True,
        )
        new_3d = Convolution3D(
            data=self._data,
            kernel_size=self._kernel_size,
            with_nans=False,
        )

        # COMPARE results
        self._print_comparison(
            name1="GenericFilter",
            name2="Frederic's STDs",
            value1=legacy.sdev,
            value2=new.sdev,
        )
        self._print_comparison(
            name1="GenericFilter with NaNs",
            name2="Frederic's STDs with NaNs",
            value1=legacy_nan.sdev,
            value2=new_nan.sdev,
        )
        self._print_comparison(
            name1="Frederic's STDs",
            name2="Frederic's STDs with nans",
            value1=new.sdev,
            value2=new_nan.sdev,
        )
        self._print_comparison(
            name1="GenericFilter",
            name2="Convolution 3D",
            value1=legacy.sdev,
            value2=new_3d.sdev,
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

    CompareSTDs(
        data=np.random.rand(36, 1024, 128).astype(np.float64) * (2**12 - 1),
        kernel_size=3,
        tolerance=1e-5,
        abs_tol=1e-8,
    )
