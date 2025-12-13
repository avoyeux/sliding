"""
Code to test if the border cases are handled the same way between different implementations.
"""
from __future__ import annotations

# IMPORTs third-party
import pytest

# IMPORTs alias
import numpy as np

# IMPORTs local
from programs.sigma_clipping import sigma_clip, FastSigmaClipping

# TYPE ANNOTATIONs
from typing import Literal



class TestBorders:
    """
    To test the values gotten from the borders when sigma clipping a given array.
    """

    @pytest.fixture(scope="class")
    def big_data(self) -> np.ndarray:
        """
        Data fixture representing a normal SPICE data cube.
        """
        return np.random.rand(36, 1024, 128).astype(np.float64) # todo change it to normal data range

    @pytest.fixture(scope="class")
    def small_data(self) -> np.ndarray:
        """
        Data fixture representing a small data cube for quicker tests.
        """
        return np.random.rand(8, 20, 12).astype(np.float64)

    @pytest.fixture(scope="class")
    def big_data_changed(self, big_data: np.ndarray) -> np.ndarray:
        """
        Data fixture of a normal SPICE data cube with some added extreme data to make sure clipping
        occurs.
        """

        data = big_data.copy()
        data[5:10, 100:200, 50: 80] = 10.
        data[15:20, 500:600, 90: 120] = 3.
        data[:4, :, 10:90] = 20.
        return data

    @pytest.fixture(scope="class")
    def small_data_changed(self, small_data: np.ndarray) -> np.ndarray:
        """
        Data fixture of a small data cube with some added extreme data to make sure clipping
        occurs.
        """

        data = small_data.copy()
        data[1:3, 5:10, 3:8] = 10.
        data[:2, :, 2:9] = 20.
        return data

    def test_inside_small_data(
            self,
            small_data: np.ndarray,
            small_data_changed: np.ndarray,
        ) -> None:
        # todo add docstring

        center_functions: list[Literal['median', 'mean']] = ['median', 'mean']
        kernel_sizes = [3, 5, 7]
        pad_mode = 'symmetric'
        pad_reflect = 'even'

        for data in [small_data, small_data_changed]:
            for kernel_size in kernel_sizes:
                for center_func in center_functions:

                    new_result = FastSigmaClipping(
                        data=data.copy(),
                        size=kernel_size,
                        sigma=2.,
                        max_iters=3,
                        center_choice=center_func,
                        masked_array=False,
                        padding_mode=pad_mode,
                        padding_reflect_type=pad_reflect,
                    ).results

                    old_result = sigma_clip(
                        data=data.copy(),
                        size=kernel_size,
                        sigma=2,
                        max_iters=3,
                        center_func=center_func,
                        masked=False,
                    )

                    # INSIDE check
                    index = kernel_size // 2
                    np.testing.assert_allclose(
                        actual=old_result[index:-index, index:-index, index:-index],
                        desired=new_result[index:-index, index:-index, index:-index],
                        rtol=1e-5,
                        atol=1e-8,
                    )

    def test_borders_small_data(
            self,
            small_data: np.ndarray,
            small_data_changed: np.ndarray,
        ) -> None:
        # todo add docstring

        center_functions: list[Literal['median', 'mean']] = ['median', 'mean']
        kernel_sizes = [3, 5, 7]
        pad_mode = 'symmetric'
        pad_reflect = 'even'

        for data in [small_data, small_data_changed]:
            for kernel_size in kernel_sizes:
                for center_func in center_functions:

                    new_result = FastSigmaClipping(
                        data=data.copy(),
                        size=kernel_size,
                        sigma=2.,
                        max_iters=3,
                        center_choice=center_func,
                        masked_array=False,
                        padding_mode=pad_mode,
                        padding_reflect_type=pad_reflect,
                    ).results

                    old_result = sigma_clip(
                        data=data.copy(),
                        size=kernel_size,
                        sigma=2,
                        max_iters=3,
                        center_func=center_func,
                        masked=False,
                    )

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

    def test_inside_big_data(
            self,
            big_data: np.ndarray,
            big_data_changed: np.ndarray,
        ) -> None:
        # todo add docstring

        center_functions: list[Literal['median', 'mean']] = ['median', 'mean']
        kernel_sizes = [3, 5, 7]
        pad_mode = 'symmetric'
        pad_reflect = 'even'

        for data in [big_data, big_data_changed]:
            for kernel_size in kernel_sizes:
                for center_func in center_functions:

                    new_result = FastSigmaClipping(
                        data=data.copy(),
                        size=kernel_size,
                        sigma=2.,
                        max_iters=3,
                        center_choice=center_func,
                        masked_array=False,
                        padding_mode=pad_mode,
                        padding_reflect_type=pad_reflect,
                    ).results

                    old_result = sigma_clip(
                        data=data.copy(),
                        size=kernel_size,
                        sigma=2,
                        max_iters=3,
                        center_func=center_func,
                        masked=False,
                    )

                    # INSIDE check
                    index = kernel_size // 2
                    np.testing.assert_allclose(
                        actual=old_result[index:-index, index:-index, index:-index],
                        desired=new_result[index:-index, index:-index, index:-index],
                        rtol=1e-5,
                        atol=1e-8,
                    )
    
    def test_borders_big_data(
            self,
            big_data: np.ndarray,
            big_data_changed: np.ndarray,
        ) -> None:
        # todo add docstring

        center_functions: list[Literal['median', 'mean']] = ['median', 'mean']
        kernel_sizes = [3, 5, 7]
        pad_mode = 'symmetric'
        pad_reflect = 'even'

        for data in [big_data, big_data_changed]:
            for kernel_size in kernel_sizes:
                for center_func in center_functions:

                    new_result = FastSigmaClipping(
                        data=data.copy(),
                        size=kernel_size,
                        sigma=2.,
                        max_iters=3,
                        center_choice=center_func,
                        masked_array=False,
                        padding_mode=pad_mode,
                        padding_reflect_type=pad_reflect,
                    ).results

                    old_result = sigma_clip(
                        data=data.copy(),
                        size=kernel_size,
                        sigma=2,
                        max_iters=3,
                        center_func=center_func,
                        masked=False,
                    )

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