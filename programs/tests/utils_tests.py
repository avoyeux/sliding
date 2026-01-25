"""
Contains utility functions that could be used in different tests.
"""
from __future__ import annotations

# IMPORTs standard
import glob
import time

# IMPORTs third-party
import pytest

# IMPORTs alias
import numpy as np
import multiprocessing as mp

# IMPORTs sub
from astropy.io import fits
from scipy.ndimage import generic_filter

# TYPE ANNOTATIONs
import queue
from typing import cast, Callable

# API public
__all__ = ['TestUtils']



class TestUtils:
    """
    Utility functions that can be used in different tests.
    """

    ADD_NANS: bool = True
    NB_PROCESSES: int = 32
    COMPARE_NANS: bool = True

    @staticmethod
    def get_filepaths() -> list[str]:
        """
        Gives the filepaths to the FITS files used in the testing.

        Returns:
            list[str]: the list of filepaths to the needed FITS files.
        """

        fits_files = glob.glob(
            '/home/voyeux-alfred/Documents/work_codes/sigma_clipping_tests/results/*L1*.fits'
        )
        return fits_files[:64]

    @staticmethod
    def open_file(filepath: str) -> np.ndarray[tuple[int, ...], np.dtype[np.float32]] | dict:
        """
        Tries to open a FITS file to get the data inside.
        If fails to do so, returns a dict containing the information from the error.

        Args:
            filepath (str): the filepath to the FITS file.

        Returns:
            np.ndarray | dict: the data inside the FITS file or a dict with the error information.
        """

        try:
            hdul = fits.open(filepath)
            data = hdul[0].data.astype(np.float32).squeeze()#type: ignore
            hdul.close()
            if TestUtils.ADD_NANS: data = TestUtils._add_NaNs(data, fraction=0.05)
            return data
        except (FileNotFoundError, OSError, IOError) as e:
            log = {
                'status': 'failure',
                'filepath': filepath,
                'error': f"File reading error: {str(e)}",
                'error_type': 'file_error',
            }
            return log

    @staticmethod
    def _add_NaNs(data: np.ndarray, fraction: float = 0.01) -> np.ndarray:
        """
        To randomly add NaN values to the given data.

        Args:
            data (np.ndarray): the data to add NaNs to.
            fraction (float): the fraction of values to turn into NaNs.

        Returns:
            np.ndarray: the data with NaNs added.
        """

        mask = np.random.rand(*data.shape) < fraction
        data[mask] = np.nan
        return data

    @staticmethod
    def compare(actual: np.ndarray, desired: np.ndarray, filepath: str) -> dict:
        """
        Asserts a comparison between all the values of the 2 given numpy arrays.
        Returns a dict with the information gotten from the assert (even if it fails).

        Args:
            actual (np.ndarray): one of the arrays to compare the values for.
            desired (np.ndarray): the second array to compare the values for.
            filepath (str): the filepath to the FITS file which was the source of the given arrays.

        Returns:
            dict: a dict containing the information from the comparison.
        """

        if not TestUtils.COMPARE_NANS:
            actual = np.where(~np.isnan(actual), actual, 0.).astype(actual.dtype)
            desired = np.where(~np.isnan(desired), desired, 0.).astype(desired.dtype)

        try:
            np.testing.assert_allclose(
                actual=actual,
                desired=desired,
                rtol=1e-5,
                atol=1e-8,
                err_msg=f"Results differ for file {filepath}",
            )
            log = {
                'status': 'success',
                'filepath': filepath,
            }
        except Exception as e:
            log = {
                'status': 'failure',
                'filepath': filepath,
                'error': str(e)
            }
        return log

    @staticmethod
    def multiprocess(
            filepaths: list[str],
            target: Callable[[queue.Queue[str | None], queue.Queue[dict]], None],
        ) -> None:
        """
        To use multiprocessing for the tests.
        The results of the tests are then caught and checked to see if any failed.

        Args:
            filepaths (list[str]): the list of filepaths to process.
            target (Callable[[queue.Queue[str | None], queue.Queue[dict]], None]): the function to
                run for each process.
        """

        start_time = time.time()
        nb_processes = min(TestUtils.NB_PROCESSES, len(filepaths))

        manager = mp.Manager()
        input_queue = manager.Queue()
        result_queue = manager.Queue()

        for filepath in filepaths: input_queue.put(filepath)
        for _ in range(nb_processes): input_queue.put(None)

        processes = cast(list[mp.Process], [None] * nb_processes)

        for i in range(nb_processes):
            p = mp.Process(
                target=target,
                args=(input_queue, result_queue),
            )
            p.start()
            processes[i] = p

        for p in processes: p.join()
        end_time = time.time()
        duration = (end_time - start_time) / 60
        print(f"\033[1;90mComparison test completed in {duration:.2f} minutes.\033[0m")

        # RESULTs collect
        errors = []
        results = []
        while not result_queue.empty():
            results.append(result_queue.get())

        if len(results) != len(filepaths):
            pytest.fail(
                f"Some files were not processed: "
                f"{len(filepaths) - len(results)} out of {len(filepaths)}"
            )

        for result in results:
            if result['status'] == 'failure':
                errors.append(f"{result['error']}")

        if errors:
            pytest.fail(f"Comparison failed for {len(errors)} files:\n" + "\n".join(errors))

    @staticmethod
    def old_implementation(
            function: str,
            data: np.ndarray,
            kernel_size: tuple[int, ...],
        ) -> np.ndarray:
        """
        The old (generic filter) implementation used to get the sliding mean and median.

        Args:
            function (str): the function type, i.e. 'std', 'mean' or 'median'.
            data (np.ndarray): the data to get the sliding values for.
            kernel_size (tuple[int, ...]): the kernel size.

        Returns:
            np.ndarray: the sliding values.
        """

        def get_numpy_function(
                data: np.ndarray,
                name: str,
            ) -> Callable[..., np.ndarray | np.floating]:
            """
            Get NaN-aware version of numpy array function if there are any NaNs in data.
            """

            if np.isnan(data).any(): name = "nan" + name
            return getattr(np, name)

        result = generic_filter(
            input=data,
            function=get_numpy_function(data, function),
            size=kernel_size,
        )
        return result
