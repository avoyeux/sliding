"""
Code to test if the border cases are handled the same way between different implementations.
"""
from __future__ import annotations

# IMPORTs standard
import time

# IMPORTs third-party
import pytest

# IMPORTs alias
import numpy as np

# IMPORTs local
from programs.standard_deviation import Convolution
from programs.sigma_clipping import sigma_clip, FastSigmaClipping
from programs.tests.sliding_mean import sliding_weighted_mean_3d
from programs.sigma_clipping.numba_functions import (
    tuple_sliding_nanmedian_3d, tuple_sliding_nanmedian_nd,
    sliding_weighted_median_3d, sliding_weighted_median_nd,
)

# TYPE ANNOTATIONs
from typing import Literal, cast, Any



class TestBorders:
    """
    To test the values gotten from the borders when sigma clipping a given array.
    """

    @pytest.fixture(scope="class")
    def big_data(self) -> np.ndarray:
        """
        Data fixture representing a normal SPICE data cube.
        """
        return np.random.rand(36, 1024, 128).astype(np.float64) * (2**12 - 1)

    @pytest.fixture(scope="class")
    def small_data(self) -> np.ndarray:
        """
        Data fixture representing a small data cube for quicker tests.
        """
        return np.random.rand(12, 20, 25, 20).astype(np.float64)

    @pytest.fixture(scope="class")
    def big_data_changed(self, big_data: np.ndarray) -> np.ndarray:
        """
        Data fixture of a normal SPICE data cube with some added extreme data to make sure clipping
        occurs.
        """

        data = big_data.copy()
        data[5:10, 100:200, 50: 80, ...] = 10. * (2**12 - 1)
        data[15:20, 500:600, 90: 120, ...] = 3.
        data[:4, :, 10:90, ...] = 20. * (2**12 - 1)
        return data

    @pytest.fixture(scope="class")
    def small_data_changed(self, small_data: np.ndarray) -> np.ndarray:
        """
        Data fixture of a small data cube with some added extreme data to make sure clipping
        occurs.
        """

        data = small_data.copy()
        data[1:3, 5:10, 3:8, ...] = 10.
        data[:2, :, 2:9, ...] = 20.
        return data

    def _get_padding(self) -> dict:
        """
        To get the corresponding np.pad parameters given the border type.
        """

        if self._borders is None:
            # ADAPTATIVE borders
            result = {
                'mode': 'constant',
                'constant_values': np.nan,
            }
        elif self._borders == 'reflect':
            result = {
                'mode': 'symmetric',
                'reflect': 'even',
            }
        elif self._borders == 'constant':
            result = {
                'mode': 'constant',
                'constant_values': 0.,
            }
        elif self._borders == 'replicate':
            result = {'mode': 'edge'}
        elif self._borders == 'wrap':
            result = {'mode': 'wrap'}
        else:
            raise ValueError(f"Unknown border type: {self._borders}")
        return result
    
    def _add_padding(self, data: np.ndarray, pad: tuple[tuple[int, int], ...]) -> np.ndarray:
        """
        To add padding to 'data' given the pad widths and the padding mode.

        Args:
            data (np.ndarray): the data to pad.
            pad (tuple[tuple[int, int], ...]): the pad widths for each dimension.

        Returns:
            np.ndarray: the padded data.
        """

        if self._padding_mode == 'constant':
            padded = np.pad(
                array=data,
                pad_width=pad,
                mode=cast(Literal['edge'], self._padding_mode),
                constant_values=self._padding_constant_values,
            )
        elif self._padding_mode in ['reflect', 'symmetric']:  # ! cannot be reflect any more
            padded = np.pad(
                array=data,
                pad_width=pad,
                mode=cast(Literal['edge'], self._padding_mode),
                reflect_type=cast(Literal['even'], self._padding_reflect_type),
            )
        else:
            padded = np.pad(
                array=data,
                pad_width=pad,
                mode=cast(Literal['edge'], self._padding_mode),
            )
        return padded

    def _borders_check(self, old_result: np.ndarray, new_result: np.ndarray) -> None:
        """
        To check if the borders of the two results are the same.

        Args:
            old_result (np.ndarray): the old result to check.
            new_result (np.ndarray): the new result to check.
        """

        # BORDERs check
        np.testing.assert_allclose(
            actual=old_result[0, :, :],
            desired=new_result[0, :, :],
            rtol=1e-5,
            atol=1e-8,
        )
        np.testing.assert_allclose(
            actual=old_result[-1, :, :],
            desired=new_result[-1, :, :],
            rtol=1e-5,
            atol=1e-8,
        )
        np.testing.assert_allclose(
            actual=old_result[:, 0, :],
            desired=new_result[:, 0, :],
            rtol=1e-5,
            atol=1e-8,
        )
        np.testing.assert_allclose(
            actual=old_result[:, -1, :],
            desired=new_result[:, -1, :],
            rtol=1e-5,
            atol=1e-8,
        )
        np.testing.assert_allclose(
            actual=old_result[:, :, 0],
            desired=new_result[:, :, 0],
            rtol=1e-5,
            atol=1e-8,
        )
        np.testing.assert_allclose(
            actual=old_result[:, :, -1],
            desired=new_result[:, :, -1],
            rtol=1e-5,
            atol=1e-8,
        )

    def test_n_dimensions(
            self,
            big_data: np.ndarray,
            big_data_changed: np.ndarray,
        ) -> None:
        """
        To compare the 3D and ND implementations of the sliding median functions.
        """

        kernel_tuple = (5, 3, 7)
        kernel_array = np.ones(kernel_tuple, dtype=big_data.dtype)
        kernel_array[2, 1, 1] = 0.
        compile_jump = tuple_sliding_nanmedian_3d(big_data, kernel_tuple)  # compile functions
        print()

        for data in [big_data, big_data_changed]:

            start = time.time()
            result_tuple_3d = tuple_sliding_nanmedian_3d(data, kernel_tuple)
            end_3d = time.time()
            result_tuple_nd = tuple_sliding_nanmedian_nd(data, kernel_tuple)
            end_nd = time.time()

            result_array_3d = sliding_weighted_median_3d(data, kernel_array)
            end_array_3d = time.time()
            result_array_nd = sliding_weighted_median_nd(data, kernel_array)
            end_array_nd = time.time()

            time_3d = end_3d - start
            time_nd = end_nd - end_3d
            time_array_3d = end_array_3d - end_nd
            time_array_nd = end_array_nd - end_array_3d

            print(f"tuple 3d: {time_3d:.2f} s")
            print(f"tuple nd: {time_nd:.2f} s")
            print(f"array 3d: {time_array_3d:.2f} s")
            print(f"array nd: {time_array_nd:.2f} s")
            print()

            # INSIDE check
            index = max(kernel_tuple) // 2
            np.testing.assert_allclose(
                actual=result_tuple_3d[index:-index, index:-index, index:-index],
                desired=result_tuple_nd[index:-index, index:-index, index:-index],
                rtol=1e-5,
                atol=1e-8,
            )
            np.testing.assert_allclose(
                actual=result_array_3d[index:-index, index:-index, index:-index],
                desired=result_array_nd[index:-index, index:-index, index:-index],
                rtol=1e-5,
                atol=1e-8,
            )

            # BORDERs check
            self._borders_check(result_tuple_3d, result_tuple_nd)
            self._borders_check(result_array_3d, result_array_nd)

    def test_padding_choices(
            self,
            big_data: np.ndarray,
            big_data_changed: np.ndarray,
        ) -> None:
        """
        To test if the different border options coincide properly between scipy, cv2 and np.pad
        implementations.
        """
        border_options = ['reflect', 'constant', 'replicate', 'wrap', None]
        kernel_size = (3, 7, 9)

        for border in border_options:
            self._borders = border
            for data in [big_data, big_data_changed]:
                kernel = np.ones(kernel_size)
                kernel[1, 1, 1] = 0.
                kernel_norm = kernel / kernel.sum()
                new_result = Convolution(
                    data=data.copy(),
                    kernel=kernel_norm,
                    borders=self._borders,
                ).result

                # PADDING
                padding_settings = self._get_padding()
                self._padding_mode = padding_settings['mode']
                self._padding_reflect_type = padding_settings.get('reflect_type', 'even')
                self._padding_constant_values = padding_settings.get('constant_values', 0.)

                pad = tuple((k // 2, k // 2) for k in kernel.shape)
                padded = self._add_padding(data, pad)
                old_result = sliding_weighted_mean_3d(padded, kernel)
                nd_result = sliding_weighted_mean_3d(padded, kernel)

                # INSIDE check
                index = max(kernel.shape) // 2
                np.testing.assert_allclose(
                    actual=old_result[index:-index, index:-index, index:-index],
                    desired=new_result[index:-index, index:-index, index:-index],
                    rtol=1e-5,
                    atol=1e-8,
                )
                np.testing.assert_allclose(
                    actual=nd_result[index:-index, index:-index, index:-index],
                    desired=new_result[index:-index, index:-index, index:-index],
                    rtol=1e-5,
                    atol=1e-8,
                )

                # BORDERs check
                self._borders_check(old_result, new_result)
                self._borders_check(nd_result, new_result)

    def test_sigma_clipping(
            self,
            big_data: np.ndarray,
            big_data_changed: np.ndarray,
        ) -> None:
        """
        To compare the full sigma clipping implementations between the old and new code.
        """
        print()
        sigma = 2
        max_iters = 1
        centers: list[Literal['mean', 'median']] = ['mean', 'median']
        kernel_size = (3, 3, 3)

        for center in centers:
            for data in [big_data, big_data_changed]:
                start = time.time()
                old_result = sigma_clip(
                    data=data.copy(),
                    size=kernel_size,
                    sigma=sigma,
                    max_iters=max_iters,
                    center_func=center,
                    masked=False,
                )
                end_old = time.time()

                new_result = FastSigmaClipping(
                    data=data.copy(),
                    kernel=kernel_size,
                    sigma=sigma,
                    max_iters=max_iters,
                    center_choice=center,
                    borders='reflect',
                    threads=None,
                    masked_array=False,
                ).results
                end_new = time.time()

                print(f"\033[1;92mOld ({center}): {end_old - start:.2f} s")
                print(f"New ({center}): {end_new - end_old:.2f} s\033[0m")

                # INSIDE check
                index = max(kernel_size) // 2
                np.testing.assert_allclose(
                    actual=old_result[index:-index, index:-index, index:-index],
                    desired=new_result[index:-index, index:-index, index:-index],
                    rtol=1e-5,
                    atol=1e-8,
                )

                # BORDERs check
                self._borders_check(old_result, new_result)
