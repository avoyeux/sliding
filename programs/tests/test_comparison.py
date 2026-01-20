"""
Code to test if the results gotten from the new fast implementation are exactly the same as the
generic filter results.
"""
from __future__ import annotations

# IMPORTs standard
import glob

# IMPORTs alias
import numpy as np
import multiprocessing as mp

# IMPORTs sub
from astropy.io import fits

# IMPORTs third-party
import pytest

# IMPORTs local
from programs.sigma_clipping import FastSigmaClipping, sigma_clip

# TYPE ANNOTATIONs
from typing import cast
import queue



class TestSigmaClipping:
    """
    To test the results gotten from the new and old sigma clipping.
    """

    @pytest.fixture(scope='class')
    def data_files(self) -> list[str]:
        """
        Gives the FITs filepaths to 
        """

        fits_files = glob.glob(
            '/home/avoyeux/Documents/work_codes/sigma_clipping_tests/results/*L1*.fits'
        )[:50]

        print(f"Found {len(fits_files)} FITs files for testing sigma clipping.")
        return fits_files

    def test_comparison(
            self,
            data_files: list[str],
        ) -> None:
        """
        Compares the results gotten from the generic filter method and the new 'fast' sigma
        clipping method.
        """

        nb_processes = min(8, len(data_files))

        manager = mp.Manager()
        input_queue = manager.Queue()
        result_queue = manager.Queue()

        for filepath in data_files: input_queue.put(filepath)
        for _ in range(nb_processes): input_queue.put(None)

        processes = cast(list[mp.Process], [None] * nb_processes)

        for i in range(nb_processes):
            p = mp.Process(
                target=self.run_process,
                args=(input_queue, result_queue),
            )
            p.start()
            processes[i] = p

        for p in processes: p.join()

        errors = []
        while not result_queue.empty():
            result = result_queue.get()
            if result['status'] == 'failure':
                errors.append(f"{result['filepath']}: {result['error']}")

        if errors:
            pytest.fail(f"Comparison failed for {len(errors)} files:\n" + "\n".join(errors))

    @staticmethod
    def run_process(input_queue: queue.Queue[str], result_queue: queue.Queue[dict]) -> None:
        # todo add docstring

        while True:
            filepath = input_queue.get()
            if filepath is None: break

            try:
                hdul = fits.open(filepath)
                data = hdul[0].data.astype(np.float64)
                hdul.close()
                print(f"Data shape: {data.shape}")

                # OLD sigma clipping
                old_result = sigma_clip(
                    data=data,
                    size=3,
                    sigma=3,
                    max_iters=2,
                    center_func='median',
                    masked=False,
                )

                # NEW sigma clipping
                fast_sigma_clipping = FastSigmaClipping(
                    data=data,
                    kernel=3,
                    center_choice='median',
                    sigma=3,
                    max_iters=2,
                    masked_array=False,
                ).results

                # COMPARISON
                np.testing.assert_allclose(
                    actual=fast_sigma_clipping,
                    desired=old_result,
                    rtol=1e-7,
                    atol=1e-10,
                    err_msg=f"Results differ for file {filepath}",
                )

                result_queue.put(
                    {'status': 'success', 'filepath': filepath}
                )
            except Exception as e:
                result_queue.put(
                    {'status': 'failure', 'filepath': filepath, 'error': str(e)}
                )
