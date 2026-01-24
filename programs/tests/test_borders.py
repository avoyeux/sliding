"""
Code to test that the border values match between cv2 and np.padding.
"""
from __future__ import annotations

# IMPORTs third-party
import pytest

# IMPORTs alias
import numpy as np

# IMPORTs local
from programs.tests.utils_tests import TestUtils
from programs.sigma_clipping import Convolution
from programs.tests.sliding_mean import sliding_weighted_mean_3d

# TYPE ANNOTATIONs
import queue



class TestBorders:
    """
    Test to compare if the border handling is the same between cv2 and np.pad.
    """

    @pytest.fixture(scope="class")
    def filepaths(self) -> list[str]:
        """
        Gives the FITs filepaths to the test FITS files.
        """

        filepaths = TestUtils.get_filepaths()
        print(f"Found {len(filepaths)} FITs files for testing borders.")
        return filepaths

    def test_borders_reflect(
            self,
            filepaths: list[str],
        ) -> None:
        """
        To test if the border cases are handled the same way between np.pad and cv2.
        """

        TestUtils.multiprocess(
            filepaths=filepaths,
            target=self._run_reflect,
        )

    def test_borders_constant(
            self,
            filepaths: list[str],
        ) -> None:
        """
        To test if the border cases are handled the same way between np.pad and cv2.
        """

        TestUtils.multiprocess(
            filepaths=filepaths,
            target=self._run_constant,
        )

    def test_borders_replicate(
            self,
            filepaths: list[str],
        ) -> None:
        """
        To test if the border cases are handled the same way between np.pad and cv2.
        """

        TestUtils.multiprocess(
            filepaths=filepaths,
            target=self._run_replicate,
        )

    def test_borders_wrap(
            self,
            filepaths: list[str],
        ) -> None:
        """
        To test if the border cases are handled the same way between np.pad and cv2.
        """

        TestUtils.multiprocess(
            filepaths=filepaths,
            target=self._run_wrap,
        )

    def test_borders_none(
            self,
            filepaths: list[str],
        ) -> None:
        """
        To test if the border cases are handled the same way between np.pad and cv2.
        """

        TestUtils.multiprocess(
            filepaths=filepaths,
            target=self._run_none,
        )

    @staticmethod
    def _run_reflect(
            input_queue: queue.Queue[str | None],
            result_queue: queue.Queue[dict],
        ) -> None:
        """
        To test the 'reflect' border option.
        """

        border = 'reflect'
        TestBorders._run_process(
            border=border,
            input_queue=input_queue,
            result_queue=result_queue,
        )

    @staticmethod
    def _run_constant(
            input_queue: queue.Queue[str | None],
            result_queue: queue.Queue[dict],
        ) -> None:
        """
        To test the 'constant' border option.
        """

        border = 'constant'
        TestBorders._run_process(
            border=border,
            input_queue=input_queue,
            result_queue=result_queue,
        )

    @staticmethod
    def _run_replicate(
            input_queue: queue.Queue[str | None],
            result_queue: queue.Queue[dict],
        ) -> None:
        """
        To test the 'replicate' border option.
        """

        border = 'replicate'
        TestBorders._run_process(
            border=border,
            input_queue=input_queue,
            result_queue=result_queue,
        )

    @staticmethod
    def _run_wrap(
            input_queue: queue.Queue[str | None],
            result_queue: queue.Queue[dict],
        ) -> None:
        """
        To test the 'wrap' border option.
        """

        border = 'wrap'
        TestBorders._run_process(
            border=border,
            input_queue=input_queue,
            result_queue=result_queue,
        )

    @staticmethod
    def _run_none(
            input_queue: queue.Queue[str | None],
            result_queue: queue.Queue[dict],
        ) -> None:
        """
        To test the 'none' border option.
        """

        border = None
        TestBorders._run_process(
            border=border,
            input_queue=input_queue,
            result_queue=result_queue,
        )

    @staticmethod
    def _run_process(
            border: str | None,
            input_queue: queue.Queue[str | None],
            result_queue: queue.Queue[dict],
        ) -> None:
        """
        Comparison test between the cv2 and np.pad border values.
        Used to make sure that my conversion in my sigma clipping implementation is the right one.

        Args:
            border (str | None): the border type.
            input_queue (queue.Queue[str | None]): the input queue with the filepaths.
            result_queue (queue.Queue[dict]): the result queue to put the results in.
        """

        kernel_size = 3

        while True:
            filepath = input_queue.get()
            if filepath is None: break

            data = TestUtils.open_file(filepath)
            if isinstance(data, dict): result_queue.put(data); continue

            # NEED 3D data (as don't have another ND sliding mean computation that uses padding)
            if data.ndim != 3:
                result_queue.put({
                    'filepath': filepath,
                    'status': 'skipped',
                    'message': f"Data ndim is {data.ndim}. Need 3D. skipping.",
                })
                continue

            # CV2
            kernel = np.ones(kernel_size)
            kernel[1, 1, 1] = 0.
            kernel_norm = kernel / kernel.sum()
            new_result = Convolution(
                data=data.copy(),
                kernel=kernel_norm,
                borders=border,#type:ignore
            ).result

            # PADDING
            pad = tuple((k // 2, k // 2) for k in kernel.shape)
            padded = TestUtils.add_padding(border, data, pad)
            old_result = sliding_weighted_mean_3d(padded, kernel)#type:ignore

            # CHECK all values
            comparison_log = TestUtils.compare(
                actual=new_result,
                desired=old_result,
                filepath=filepath,
            )
            result_queue.put(comparison_log)
