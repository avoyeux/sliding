"""
To check the duration of the different implementations.
To be run when the pytest pass successfully as this only checks the time taken.
This is run on the same data for all so that loading times and other factors don't play as
much of a role for the computation time.
"""
from __future__ import annotations

# IMPORTs standard
import time

# IMPORTs alias
import numpy as np
import multiprocessing as mp

# IMPORTs sub
from multiprocessing import shared_memory

# IMPORTs local
from programs.tests.utils_tests import TestUtils
from programs.sigma_clipping import (
    FastStandardDeviation, FastSigmaClipping, sigma_clip, SlidingMedian, SlidingMean,
)

# TYPE ANNOTATIONs
from typing import Callable, cast
from multiprocessing.sharedctypes import Synchronized
from multiprocessing.synchronize import Lock as LockType
type CounterType = Synchronized[int]
type SharedMemoryType = shared_memory.SharedMemory



class CheckTimes:
    """
    To check the running times between the old and new implementations.
    """

    def __init__(self, processes: int, jobs: int) -> None:

        self._jobs = jobs
        self._processes = processes

        print(f"Time checks when running {self._jobs} jobs.")

    def run_standard_deviation(self) -> None:
        """
        Runs the standard deviation time checks.
        """

        print(f"\033[1;34mChecking standard deviation...\033[0m")

        self._multiprocess(
            name='New standard deviation',
            target=self._worker_new_std,
        )
        self._multiprocess(
            name='Old standard deviation',
            target=self._worker_old_std,
        )

    def run_mean(self) -> None:
        """
        Runs the sliding mean time checks.
        """

        print(f"\033[1;34mChecking mean...\033[0m")

        self._multiprocess(
            name='New mean',
            target=self._worker_new_mean,
        )
        self._multiprocess(
            name='Old mean',
            target=self._worker_old_mean,
        )

    def run_median(self) -> None:
        """
        Runs the sliding median time checks.
        """

        print(f"\033[1;34mChecking median...\033[0m")

        self._multiprocess(
            name='New median',
            target=self._worker_new_median,
        )
        self._multiprocess(
            name='Old median',
            target=self._worker_old_median,
        )

    def run_sigma_clipping_mean(self) -> None:
        """
        Runs the sigma clipping time checks.
        """

        print(f"\033[1;34mChecking sigma clipping mean...\033[0m")

        self._multiprocess(
            name='New sigma clipping mean',
            target=self._worker_new_sigma_clipping_mean,
        )
        self._multiprocess(
            name='Old sigma clipping mean',
            target=self._worker_old_sigma_clipping_mean,
        )

    def run_sigma_clipping_median(self) -> None:
        """
        Runs the sigma clipping time checks.
        """

        print(f"\033[1;34mChecking sigma clipping median...\033[0m")

        self._multiprocess(
            name='New sigma clipping median',
            target=self._worker_new_sigma_clipping_median,
        )
        self._multiprocess(
            name='Old sigma clipping median',
            target=self._worker_old_sigma_clipping_median,
        )

    @staticmethod
    def _fake_array() -> np.ndarray:

        data = np.random.rand(36, 1024, 128).astype(np.float64)
        data[10:15, 100:200, 50:75] = 1.3
        data[7:, 40:60, 70:] = 1.7

        data = TestUtils.add_NaNs(data, fraction=0.05)
        return data

    def _multiprocess(
            self,
            name: str,
            target: Callable[[CounterType, LockType, dict], None],
        ) -> None:
        """
        To multiprocess the computation.

        Args:
            name (str): the name to add in the print.
            target (Callable[[CounterType, LockType, dict], None]): the function to run for each
                process.
        """

        # DATA
        array = self._fake_array()
        if isinstance(array, dict): raise RuntimeError("File not found"); return

        # SHARED MEMORY
        shm = shared_memory.SharedMemory(create=True, size=array.nbytes)
        info = {
            'name': shm.name,
            'shape': array.shape,
            'dtype': array.dtype,
        }
        shared_array = np.ndarray(array.shape, dtype=array.dtype, buffer=shm.buf)
        np.copyto(shared_array, array)
        shm.close()

        lock = mp.Lock()
        counter: CounterType = mp.Value('i', self._jobs)
        processes: list[mp.Process] = cast(list[mp.Process], [None] * self._processes)

        start_time = time.time()
        for i in range(self._processes):
            p = mp.Process(
                target=target,
                args=(counter, lock, info),
            )
            p.start()
            processes[i] = p
        for p in processes: p.join()
        duration = (time.time() - start_time) / 60

        shm.unlink()
        print(f"\033[1;90m{name} completed in {duration:.2f} minutes.\033[0m")

    @staticmethod
    def _open_shared_memory(shared: dict) -> tuple[SharedMemoryType, np.ndarray]:

        shm = shared_memory.SharedMemory(name=shared['name'])
        array = np.ndarray(
            shape=shared['shape'],
            dtype=shared['dtype'],
            buffer=shm.buf,
        )
        return shm, array

    @staticmethod
    def _worker_new_std(counter: CounterType, lock: LockType, shared: dict) -> None:

        shm, data = CheckTimes._open_shared_memory(shared)
        kernel = (3,) * data.ndim

        while True:
            with lock:
                if counter.value <= 0: break
                counter.value -= 1

            _ = FastStandardDeviation(
                data=data,
                kernel=kernel,
                borders='reflect',
            ).standard_deviation
        shm.close()

    @staticmethod
    def _worker_old_std(counter: CounterType, lock: LockType, shared: dict) -> None:

        shm, data = CheckTimes._open_shared_memory(shared)
        kernel = (3,) * data.ndim

        while True:
            with lock:
                if counter.value <= 0: break
                counter.value -= 1

            _ = TestUtils.old_implementation(
                function='std',
                data=data,
                kernel_size=kernel,
            )
        shm.close()

    @staticmethod
    def _worker_new_mean(counter: CounterType, lock: LockType, shared: dict) -> None:

        shm, data = CheckTimes._open_shared_memory(shared)
        kernel = (3,) * data.ndim

        while True:
            with lock:
                if counter.value <= 0: break
                counter.value -= 1

            _ = SlidingMean(
                data=data,
                kernel=kernel,
                borders='reflect',
                threads=1,
            ).mean
        shm.close()

    @staticmethod
    def _worker_old_mean(counter: CounterType, lock: LockType, shared: dict) -> None:

        shm, data = CheckTimes._open_shared_memory(shared)
        kernel = (3,) * data.ndim

        while True:
            with lock:
                if counter.value <= 0: break
                counter.value -= 1

            _ = TestUtils.old_implementation(
                function='mean',
                data=data,
                kernel_size=kernel,
            )
        shm.close()

    @staticmethod
    def _worker_new_median(counter: CounterType, lock: LockType, shared: dict) -> None:

        shm, data = CheckTimes._open_shared_memory(shared)
        kernel = (3,) * data.ndim

        while True:
            with lock:
                if counter.value <= 0: break
                counter.value -= 1

            _ = SlidingMedian(
                data=data,
                kernel=kernel,
                borders='reflect',
                threads=1,
            ).median
        shm.close()

    @staticmethod
    def _worker_old_median(counter: CounterType, lock: LockType, shared: dict) -> None:

        shm, data = CheckTimes._open_shared_memory(shared)
        kernel = (3,) * data.ndim

        while True:
            with lock:
                if counter.value <= 0: break
                counter.value -= 1

            _ = TestUtils.old_implementation(
                function='median',
                data=data,
                kernel_size=kernel,
            )
        shm.close()

    @staticmethod
    def _worker_new_sigma_clipping_mean(counter: CounterType, lock: LockType, shared: dict) -> None:

        shm, data = CheckTimes._open_shared_memory(shared)
        kernel = 5

        while True:
            with lock:
                if counter.value <= 0: break
                counter.value -= 1

            _ = FastSigmaClipping(
                data=data,
                kernel=kernel,
                center_choice='mean',
                sigma=2.,
                max_iters=3,
                masked_array=False,
            ).results
        shm.close()

    @staticmethod
    def _worker_old_sigma_clipping_mean(counter: CounterType, lock: LockType, shared: dict) -> None:

        shm, data = CheckTimes._open_shared_memory(shared)
        kernel = 5

        while True:
            with lock:
                if counter.value <= 0: break
                counter.value -= 1

            _ = sigma_clip(
                data=data,
                size=kernel,
                sigma=2.,
                max_iters=3,
                center_func='mean',
                masked=False,
            )
        shm.close()

    @staticmethod
    def _worker_new_sigma_clipping_median(counter: CounterType, lock: LockType, shared: dict) -> None:

        shm, data = CheckTimes._open_shared_memory(shared)
        kernel = 5

        while True:
            with lock:
                if counter.value <= 0: break
                counter.value -= 1

            _ = FastSigmaClipping(
                data=data,
                kernel=kernel,
                center_choice='median',
                sigma=2.,
                max_iters=3,
                masked_array=False,
            ).results
        shm.close()

    @staticmethod
    def _worker_old_sigma_clipping_median(counter: CounterType, lock: LockType, shared: dict) -> None:

        shm, data = CheckTimes._open_shared_memory(shared)
        kernel = 5

        while True:
            with lock:
                if counter.value <= 0: break
                counter.value -= 1

            _ = sigma_clip(
                data=data,
                size=kernel,
                sigma=2.,
                max_iters=3,
                center_func='median',
                masked=False,
            )
        shm.close()



if __name__ == '__main__':
    checker = CheckTimes(processes=94, jobs=750)

    start_time = time.time()
    print("\033[1;32mStarting duration checks...\033[0m")
    # checker.run_standard_deviation()
    # checker.run_mean()
    # checker.run_median()
    checker.run_sigma_clipping_mean()
    checker.run_sigma_clipping_median()
    end_time = time.time()
    total_duration = (end_time - start_time) / 60
    print(f"\033[1;90mAll duration checks completed in {total_duration:.2f} minutes.\033[0m")
