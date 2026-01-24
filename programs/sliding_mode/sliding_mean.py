"""
Code to compute the sliding mean given a kernel.
"""
from __future__ import annotations

# IMPORTs alias
import numpy as np
import numpy.typing as npt

# IMPORTs local
from programs.standard_deviation import BorderType, Convolution  # ! need to change the path to this code

# TYPE ANNOTATIONs
from typing import cast, Any, overload, Protocol, TypeVar
type Array[D: np.floating] = npt.NDArray[D]
D_co = TypeVar("D_co", bound=np.floating, covariant=True)
D_inv = TypeVar("D_inv", bound=np.floating, covariant=False)
class _Single(Protocol[D_co]): _mean: npt.NDArray[D_co]
class _Multi(Protocol[D_inv]): _mean: list[npt.NDArray[D_inv]]

# API public
__all__ = ["SlidingMean"]



class SlidingMean[D: np.floating[Any]]:

    @overload
    def __init__(
        self,
        data: Array[D],
        kernel: Array[D],
        borders: BorderType = "reflect",
        threads: int = 1,
    ) -> None: ...

    @overload
    def __init__(
        self,
        data: list[Array[D]],
        kernel: Array[D],
        borders: BorderType = "reflect",
        threads: int = 1,
    ) -> None: ...

    def __init__(
        self,
        data: Array[D] | list[Array[D]],
        kernel: Array[D],
        borders: BorderType = "reflect",
        threads: int = 1,
    ) -> None:
        # todo add docstring

        # ? do i need to normalize the kernel, don't think so no ?

        self._data = cast(list[Array[D]], data if isinstance(data, list) else [data])
        self._kernel = kernel
        self._borders = borders
        self._threads = threads

        # COMPUTE
        self._mean = self._sliding_mean()

    @overload
    def sliding_mean(self: _Single[D]) -> Array[D]: ...

    @overload
    def sliding_mean(self: _Multi[D]) -> list[Array[D]]: ...

    def sliding_mean(self) -> Array[D] | list[Array[D]]:
        # todo add docstring

        if len(self._mean) == 1: return self._mean[0]
        return self._mean

    def _get_not_nans_count(self) -> tuple[np.ndarray[tuple[int, ...], np.dtype[np.bool_]], Array[D]]:
        # todo add docstring

        # ! if I normalise the kernel, no need for this if there is no NaN

        # TYPE CHECKER complains
        self._borders = cast(BorderType, self._borders)

        # NaN handling
        valid_mask = ~np.isnan(self._data[0])  # even if no Nans need it for border effects

        # COUNT NaN
        count = Convolution(
            data=valid_mask.astype(self._data[0].dtype),
            kernel=np.ones(self._kernel.shape, dtype=self._data[0].dtype),
            borders=self._borders,
            threads=self._threads,
        ).result
        return valid_mask, count

    def _sliding_mean(self) -> Array[D] | list[Array[D]]:
        # todo add docstring

        # TYPE CHECKER complains
        self._borders = cast(BorderType, self._borders)

        # NaN handling
        nan_mask, count = self._get_not_nans_count()

        # MEMORY pre-allocation
        means: list[Array[D]] = cast(list[Array[D]], [None] * len(self._data))

        for i, data in enumerate(self._data):

            # NaN handling
            if nan_mask is not None:
                data_filled = np.where(nan_mask, data, 0.).astype(data.dtype)
            else:
                data_filled = data

            # SUM
            sum_values = Convolution(
                data=data_filled,
                kernel=self._kernel,
                borders=self._borders,
                threads=self._threads,
            ).result
            with np.errstate(divide='ignore', invalid='ignore'):
                means[i] = np.where(count > 0, sum_values / count, 0.0).astype(data.dtype)
        return means
    


if __name__ == "__main__":
    data = np.ones((10, 10), dtype=np.float32)
    instance = SlidingMean(
        data=data,
        kernel=np.ones((3, 3), dtype=np.float32),
        borders='reflect',
        threads=1,
    )
    mean = instance.sliding_mean()
