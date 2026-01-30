"""
Code to test if the results gotten from the new fast implementation are exactly the same as the
generic filter results.
"""
from __future__ import annotations

# IMPORTs third-party
import pytest

# IMPORTs local
from tests.utils import TestUtils, sigma_clip
from sliding import SlidingSigmaClipping

# TYPE ANNOTATIONs
import queue



class TestSigmaClipping:
    """
    To test the results gotten from the new and old sigma clipping.
    """

    @pytest.fixture(scope='class')
    def filepaths(self) -> list[str]:
        """
        Gives the FITs filepaths to the test FITS files.
        """

        filepaths = TestUtils.get_filepaths()
        print(f"Found {len(filepaths)} FITs files for testing sigma clipping.")
        return filepaths

    def test_clipping_mean(
            self,
            filepaths: list[str],
        ) -> None:
        """
        Compares the results gotten from the generic filter method and the new 'fast' sigma
        clipping method.
        Uses multiprocessing to speed up the comparisons.
        """

        TestUtils.multiprocess(
            filepaths=filepaths,
            target=self._run_mean,
        )

    def test_clipping_median(
            self,
            filepaths: list[str],
        ) -> None:
        """
        Compares the results gotten from the generic filter method and the new 'fast' sigma
        clipping method.
        Uses multiprocessing to speed up the comparisons.
        """

        TestUtils.multiprocess(
            filepaths=filepaths,
            target=self._run_median,
        )

    @staticmethod
    def _run_mean(
            input_queue: queue.Queue[str | None],
            result_queue: queue.Queue[dict],
        ) -> None:
        # todo add docstring

        TestSigmaClipping._run_process(
            input_queue=input_queue,
            result_queue=result_queue,
            center='mean',
        )

    @staticmethod
    def _run_median(
            input_queue: queue.Queue[str | None],
            result_queue: queue.Queue[dict],
        ) -> None:
        # todo add docstring

        TestSigmaClipping._run_process(
            input_queue=input_queue,
            result_queue=result_queue,
            center='median',
        )

    @staticmethod
    def _run_process(
            input_queue: queue.Queue[str | None],
            result_queue: queue.Queue[dict],
            center: str,
        ) -> None:
        """
        Old and new mean implementation test for sigma clipping.

        Args:
            input_queue (queue.Queue[str | None]): the input queue with the filepaths.
            result_queue (queue.Queue[dict]): the result queue to put the results in.
            center (str): the center function to use ('mean' or 'median').
        """

        while True:
            filepath = input_queue.get()
            if filepath is None: break

            data = TestUtils.open_file(filepath)
            if isinstance(data, dict): result_queue.put(data); continue

            # OLD sigma clipping
            old_result = sigma_clip(
                data=data,
                size=3,
                sigma=2.,
                max_iters=3,
                center_func=center,#type:ignore
                masked=False,
            )

            # NEW sigma clipping
            fast_sigma_clipping = SlidingSigmaClipping(
                data=data,
                kernel=3,
                center_choice=center,#type:ignore
                sigma=2.,
                max_iters=3,
                masked_array=False,
            ).results

            # COMPARISON
            comparison_log = TestUtils.compare(
                actual=old_result,
                desired=fast_sigma_clipping,
                filepath=filepath,
            )
            result_queue.put(comparison_log)
