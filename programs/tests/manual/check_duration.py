"""
To check the duration of the different implementations.
To be run when the pytest pass successfully as this only checks the time taken.
"""
from __future__ import annotations

# IMPORTs standard
import time

# IMPORTs alias
import multiprocessing as mp

# IMPORTs local
from programs.tests.utils_tests import TestUtils
from programs.sigma_clipping import FastStandardDeviation, FastSigmaClipping, sigma_clip

# TYPE ANNOTATIONs
import queue
from typing import Callable, cast



class CheckTimes:
    # todo add docstring

    def __init__(self, processes: int) -> None:
        # todo add docstring

        self._filepaths = TestUtils.get_filepaths()
        self._processes = min(processes, len(self._filepaths))

        print(f"Time checks for {len(self._filepaths)} files.")

    def run_standard_deviation(self) -> None:
        # todo add docstring

        print(f"\033[1;34mChecking standard deviation...\033[0m")

        # NEW
        self._multiprocess(
            name='New standard deviation',
            target=self._worker_new_std,
        )

        # OLD
        self._multiprocess(
            name='Old standard deviation',
            target=self._worker_old_std,
        )

    def run_sigma_clipping(self) -> None:
        # todo add docstring

        print(f"\033[1;34mChecking sigma clipping...\033[0m")

        # NEW
        self._multiprocess(
            name='New sigma clipping',
            target=self._worker_new_sigma_clipping,
        )

        # OLD
        self._multiprocess(
            name='Old sigma clipping',
            target=self._worker_old_sigma_clipping,
        )

    def _multiprocess(
            self,
            name: str,
            target: Callable[[queue.Queue[str | None]], None],
        ) -> None:
        # todo add docstring

        manager = mp.Manager()
        filepath_queue: queue.Queue[str | None] = manager.Queue()

        # FILL queue
        for filepath in self._filepaths: filepath_queue.put(filepath)
        for _ in range(self._processes): filepath_queue.put(None)

        # TIMER
        start_time = time.time()

        processes = cast(list[mp.Process], [None] * self._processes)
        for i in range(self._processes):
            p = mp.Process(
                target=target,
                args=(filepath_queue,),
            )
            p.start()
            processes[i] = p
        for p in processes: p.join()

        end_time = time.time()
        duration = (end_time - start_time) / 60
        print(f"\033[1;90m{name} completed in {duration:.2f} minutes.\033[0m")

    @staticmethod
    def _worker_new_std(
            filepath_queue: queue.Queue[str | None],
        ) -> None:
        # todo add docstring

        while True:
            filepath = filepath_queue.get()
            if filepath is None: break

            data = TestUtils.open_file(filepath)
            if isinstance(data, dict): print('file not opened:', filepath, flush=True); continue

            kernel = (3,) * data.ndim

            _ = FastStandardDeviation(
                data=data,
                kernel=kernel,
                borders='reflect',
                threads=1,
            ).sdev

    @staticmethod
    def _worker_old_std(
            filepath_queue: queue.Queue[str | None],
        ) -> None:
        # todo add docstring

        while True:
            filepath = filepath_queue.get()
            if filepath is None: break

            data = TestUtils.open_file(filepath)
            if isinstance(data, dict): print('file not opened:', filepath, flush=True); continue

            kernel = (3,) * data.ndim

            _ = TestUtils.old_implementation(
                function='std',
                data=data,
                kernel_size=kernel,
            )

    @staticmethod
    def _worker_new_sigma_clipping(
            filepath_queue: queue.Queue[str | None],
        ) -> None:
        # todo add docstring

        while True:
            filepath = filepath_queue.get()
            if filepath is None: break

            data = TestUtils.open_file(filepath)
            if isinstance(data, dict): print('file not opened:', filepath, flush=True); continue

            _ = FastSigmaClipping(
                data=data,
                kernel=5,
                center_choice='median',
                sigma=2.,
                max_iters=3,
                masked_array=False,
            ).results

    @staticmethod
    def _worker_old_sigma_clipping(
            filepath_queue: queue.Queue[str | None],
        ) -> None:
        # todo add docstring

        while True:
            filepath = filepath_queue.get()
            if filepath is None: break

            data = TestUtils.open_file(filepath)
            if isinstance(data, dict): print('file not opened:', filepath, flush=True); continue

            _ = sigma_clip(
                data=data,
                size=5,
                sigma=2.,
                max_iters=3,
                center_func='median',
                masked=False,
            )



if __name__ == '__main__':
    checker = CheckTimes(processes=32)
    # checker.run_standard_deviation()
    checker.run_sigma_clipping()
