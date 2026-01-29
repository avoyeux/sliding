"""
Code to test that the border values match between cv2 and np.padding.
"""
from __future__ import annotations

# IMPORTs third-party
import pytest

# IMPORTs alias
import numpy as np

# IMPORTs sub
from scipy.ndimage import generic_filter

# IMPORTs local
from tests.utils import TestUtils
from sliding import SlidingMean  # ? change it to .mean from standard deviation ?

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

    # def test_borders_none(
    #         self,
    #         filepaths: list[str],
    #     ) -> None:
    #     """
    #     To test if the border cases are handled the same way between np.pad and cv2.
    #     """

    #     TestUtils.multiprocess(
    #         filepaths=filepaths,
    #         target=self._run_none,
    #     )

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
    def _run_constant(  # ! fails with NaNs but doesn't otherwise... no clue how or why
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

    # @staticmethod
    # def _run_none(
    #         input_queue: queue.Queue[str | None],
    #         result_queue: queue.Queue[dict],
    #     ) -> None:
    #     """
    #     To test the 'none' border option.
    #     """

    #     border = None
    #     TestBorders._run_process(
    #         border=border,
    #         input_queue=input_queue,
    #         result_queue=result_queue,
    #     )

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

        kernel_size = 5
        assert border is not None

        while True:
            filepath = input_queue.get()
            if filepath is None: break

            data = TestUtils.open_file(filepath)
            if isinstance(data, dict): result_queue.put(data); continue

            kernel = (kernel_size,) * data.ndim

            # CV2
            new_result = SlidingMean(
                data=data.copy(),
                kernel=kernel,
                borders=border,#type:ignore
                threads=1,
            ).mean

            mode = 'nearest' if border == 'replicate' else border
            old_result = generic_filter(
                input=data,
                function=lambda x: np.nanmean(x),
                size=kernel,
                mode=mode,
            )

            # CHECK all values
            comparison_log = TestUtils.compare(
                actual=new_result,
                desired=old_result,
                filepath=filepath,
            )
            result_queue.put(comparison_log)
