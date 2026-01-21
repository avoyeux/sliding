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

        filepaths = TestUtils.get_filepaths()[16:120]
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

            # COMPARISON
            comparison_log = TestUtils.compare(
                actual=new_standard_deviation,
                desired=old_standard_deviation,
                filepath=filepath,
            )
            result_queue.put(comparison_log)

    # def test_padding_choices(  # ! I do need this inside a given test class
    #         self,
    #         filepaths: list[str],
    #     ) -> None:
    #     """
    #     To test if the different border options coincide properly between scipy, cv2 and np.pad
    #     implementations.
    #     """

    #     border_options = ['reflect', 'constant', 'replicate', 'wrap', None]
    #     kernel_size = (3, 7, 9)

    #     for border in border_options:
    #         kernel = np.ones(kernel_size)
    #         kernel[1, 1, 1] = 0.
    #         kernel_norm = kernel / kernel.sum()
    #         new_result = Convolution(
    #             data=data.copy(),
    #             kernel=kernel_norm,
    #             borders=border,
    #         ).result

    #         # PADDING
    #         pad = tuple((k // 2, k // 2) for k in kernel.shape)
    #         padded = TestUtils.add_padding(border, data, pad)
    #         old_result = sliding_weighted_mean_3d(padded, kernel)

    #         # INSIDE check
    #         index = max(kernel.shape) // 2
    #         np.testing.assert_allclose(
    #             actual=old_result[index:-index, index:-index, index:-index],
    #             desired=new_result[index:-index, index:-index, index:-index],
    #             rtol=1e-5,
    #             atol=1e-8,
    #         )

    #         # BORDERs check
    #         self._borders_check(old_result, new_result)

    # @staticmethod
    # def _borders_check(old_result: np.ndarray, new_result: np.ndarray) -> None:
    #     """
    #     To check if the borders of the two results are the same.

    #     Args:
    #         old_result (np.ndarray): the old result to check.
    #         new_result (np.ndarray): the new result to check.
    #     ? should I change this to use the TestUtils.compare function instead ?
    #     """

    #     # BORDERs check
    #     np.testing.assert_allclose(
    #         actual=old_result[0, :, :],
    #         desired=new_result[0, :, :],
    #         rtol=1e-5,
    #         atol=1e-8,
    #     )
    #     np.testing.assert_allclose(
    #         actual=old_result[-1, :, :],
    #         desired=new_result[-1, :, :],
    #         rtol=1e-5,
    #         atol=1e-8,
    #     )
    #     np.testing.assert_allclose(
    #         actual=old_result[:, 0, :],
    #         desired=new_result[:, 0, :],
    #         rtol=1e-5,
    #         atol=1e-8,
    #     )
    #     np.testing.assert_allclose(
    #         actual=old_result[:, -1, :],
    #         desired=new_result[:, -1, :],
    #         rtol=1e-5,
    #         atol=1e-8,
    #     )
    #     np.testing.assert_allclose(
    #         actual=old_result[:, :, 0],
    #         desired=new_result[:, :, 0],
    #         rtol=1e-5,
    #         atol=1e-8,
    #     )
    #     np.testing.assert_allclose(
    #         actual=old_result[:, :, -1],
    #         desired=new_result[:, :, -1],
    #         rtol=1e-5,
    #         atol=1e-8,
    #     )
