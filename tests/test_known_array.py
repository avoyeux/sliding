"""
To test the new implementations on data for which I have manually calculated the expected results.
"""
from __future__ import annotations

# IMPORTs
import pytest

# IMPORTs alias
import numpy as np

# IMPORTs local
from sliding import SlidingMean, SlidingMedian, SlidingStandardDeviation



class TestKnownArrays:
    """
    To test the sliding computations on an array with known expected results.
    Also tests the None borders option.
    """

    @pytest.fixture(scope='class')
    def data_2d(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Provides a 2D array for which the expected results are known.

        Returns:
            tuple[np.ndarray, np.ndarray]: the kernel and data to use for the tests.
        """

        # KERNEL pre-computed
        kernel = np.ones((3, 3), dtype=np.float32)
        kernel[1, 1] = 0.

        # DATA pre-computed
        computed_data = np.array([
            [np.nan, 3, 1, 0],
            [5, 2, np.nan, 4],
            [1, np.nan, 5, 3],
            [1, 0, 3, 4],
        ], dtype=np.float32)
        return kernel, computed_data

    def _known_mean_constant(self) -> np.ndarray:
        """
        Gives the expected sliding mean results when the borders are set to 'constant'.

        Returns:
            np.ndarray: the expected sliding mean results.
        """

        mean = np.array([
            [1.25, 4/3, 9/7, 5/7],
            [1., 3., 18/7, 9/7],
            [8/7, 17/7, 8/3, 16/7],
            [1/7, 10/7, 12/7, 11/8],
        ], dtype=np.float32)
        return mean

    def _know_mean_none(self) -> np.ndarray:
        """
        Gives the expected sliding mean results when the borders are set to None.

        Returns:
            np.ndarray: the expected sliding mean results.
        """

        mean = np.array([
            [10/3, 8/3, 9/4, 5/2],
            [2., 3., 18/7, 9/4],
            [2., 17/7, 8/3, 4.],
            [.5, 2.5, 3., 11/3],
        ], dtype=np.float32)
        return mean

    def _known_median_constant(self) -> np.ndarray:
        """
        Gives the expected sliding median results when the borders are set to 'constant'.

        Returns:
            np.ndarray: the expected sliding median results.
        """

        median = np.array([
                [0., .5, 0., 0.],
                [.5, 3., 3., 0.],
                [0., 2., 3., 3.],
                [0., 1., 0., 0.],
            ], dtype=np.float32)
        return median

    def _known_median_none(self) -> np.ndarray:
        """
        Gives the expected sliding median results when the borders are set to None.

        Returns:
            np.ndarray: the expected sliding median results.
        """

        median = np.array([
            [3., 2., 2.5, 2.5],
            [2., 3., 3., 2.],
            [1.5, 2., 3., 4.],
            [.5, 2., 3.5, 3.],
        ], dtype=np.float32)
        return median

    def _known_std_constant(self) -> np.ndarray:
        """
        Gives the expected sliding standard deviation results when the borders are set to
        'constant'.

        Returns:
            np.ndarray: the expected sliding standard deviation results.
        """

        std_0_0 = np.std([0, 0, 0, 0, 0, 2, 3, 5], dtype=np.float32)
        std_0_1 = np.std([0, 0, 0, 1, 2, 5], dtype=np.float32)
        std_0_2 = np.std([0, 0, 0, 0, 2, 3, 4], dtype=np.float32)
        std_0_3 = np.std([0, 0, 0, 0, 0, 1, 4], dtype=np.float32)
        std_1_0 = np.std([0, 0, 0, 1, 2, 3], dtype=np.float32)
        std_1_1 = np.std([1, 1, 3, 5, 5], dtype=np.float32)
        std_1_2 = np.std([0, 1, 2, 3, 3, 4, 5], dtype=np.float32)
        std_1_3 = np.std([0, 0, 0, 0, 1, 3, 5], dtype=np.float32)
        std_2_0 = np.std([0, 0, 0, 0, 1, 2, 5], dtype=np.float32)
        std_2_1 = np.std([0, 1, 1, 2, 3, 5, 5], dtype=np.float32)
        std_2_2 = np.std([0, 2, 3, 3, 4, 4], dtype=np.float32)
        std_2_3 = np.std([0, 0, 0, 3, 4, 4, 5], dtype=np.float32)
        std_3_0 = np.std([0, 0, 0, 0, 0, 0, 1], dtype=np.float32)
        std_3_1 = np.std([0, 0, 0, 1, 1, 3, 5], dtype=np.float32)
        std_3_2 = np.std([0, 0, 0, 0, 3, 4, 5], dtype=np.float32)
        std_3_3 = np.std([0, 0, 0, 0, 0, 3, 3, 5], dtype=np.float32)

        std = np.array([
            [std_0_0, std_0_1, std_0_2, std_0_3],
            [std_1_0, std_1_1, std_1_2, std_1_3],
            [std_2_0, std_2_1, std_2_2, std_2_3],
            [std_3_0, std_3_1, std_3_2, std_3_3],
        ], dtype=np.float32)
        return std

    def _known_std_none(self) -> np.ndarray:
        """
        Gives the expected sliding standard deviation results when the borders are set to None.

        Returns:
            np.ndarray: the expected sliding standard deviation results.
        """

        std_0_0 = np.std([2, 3, 5], dtype=np.float32)
        std_0_1 = np.std([1, 2, 5], dtype=np.float32)
        std_0_2 = np.std([0, 2, 3, 4], dtype=np.float32)
        std_0_3 = np.std([1, 4], dtype=np.float32)
        std_1_0 = np.std([1, 2, 3], dtype=np.float32)
        std_1_1 = np.std([1, 1, 3, 5, 5], dtype=np.float32)
        std_1_2 = np.std([0, 1, 2, 3, 3, 4, 5], dtype=np.float32)
        std_1_3 = np.std([0, 1, 3, 5], dtype=np.float32)
        std_2_0 = np.std([0, 1, 2, 5], dtype=np.float32)
        std_2_1 = np.std([0, 1, 1, 2, 3, 5, 5], dtype=np.float32)
        std_2_2 = np.std([0, 2, 3, 3, 4, 4], dtype=np.float32)
        std_2_3 = np.std([3, 4, 4, 5], dtype=np.float32)
        std_3_0 = np.std([0, 1], dtype=np.float32)
        std_3_1 = np.std([1, 1, 3, 5], dtype=np.float32)
        std_3_2 = np.std([0, 3, 4, 5], dtype=np.float32)
        std_3_3 = np.std([3, 3, 5], dtype=np.float32)

        std = np.array([
            [std_0_0, std_0_1, std_0_2, std_0_3],
            [std_1_0, std_1_1, std_1_2, std_1_3],
            [std_2_0, std_2_1, std_2_2, std_2_3],
            [std_3_0, std_3_1, std_3_2, std_3_3],
        ], dtype=np.float32)
        return std

    def test_constant_mean_sliding(
            self,
            data_2d: tuple[np.ndarray, np.ndarray],
        ) -> None:
        """
        Test the SlidingMean function when borders are set to 'constant'.

        Args:
            data_2d (tuple[np.ndarray, np.ndarray]): the kernel and data to test for.
        """

        kernel, data = data_2d
        expected = self._known_mean_constant()

        result = SlidingMean(
            data=data,
            kernel=kernel,
            borders='constant',
        ).mean

        np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-5)

    def test_none_mean_sliding(
            self,
            data_2d: tuple[np.ndarray, np.ndarray],
        ) -> None:
        """
        Test the SlidingMean function when borders are set to None.

        Args:
            data_2d (tuple[np.ndarray, np.ndarray]): the kernel and data to test for.
        """

        kernel, data = data_2d
        expected = self._know_mean_none()

        result = SlidingMean(
            data=data,
            kernel=kernel,
            borders=None,
        ).mean

        np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-5)

    def test_constant_mean_std(
            self,
            data_2d: tuple[np.ndarray, np.ndarray],
        ) -> None:
        """
        Tests the mean of the SlidingStandardDeviation class when borders are set to 'constant'.

        Args:
            data_2d (tuple[np.ndarray, np.ndarray]): the kernel and data to test for.
        """

        kernel, data = data_2d
        expected = self._known_mean_constant()

        result = SlidingStandardDeviation(
            data=data,
            kernel=kernel,
            borders='constant',
        ).mean

        np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-5)

    def test_none_mean_std(
            self,
            data_2d: tuple[np.ndarray, np.ndarray],
        ) -> None:
        """
        Tests the mean of the SlidingStandardDeviation class when borders are set to None.

        Args:
            data_2d (tuple[np.ndarray, np.ndarray]): the kernel and data to test for.
        """

        kernel, data = data_2d
        expected = self._know_mean_none()

        result = SlidingStandardDeviation(
            data=data,
            kernel=kernel,
            borders=None,
        ).mean

        np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-5)

    def test_constant_median_sliding(
            self,
            data_2d: tuple[np.ndarray, np.ndarray],
        ) -> None:
        """
        Tests the SlidingStandardDeviation class when borders are set to 'constant'.

        Args:
            data_2d (tuple[np.ndarray, np.ndarray]): the kernel and data to test for.
        """

        kernel, data = data_2d
        expected = self._known_median_constant()

        result = SlidingMedian(
            data=data,
            kernel=kernel,
            borders='constant',
        ).median

        np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-5)

    def test_none_median_sliding(
            self,
            data_2d: tuple[np.ndarray, np.ndarray],
        ) -> None:
        """
        Tests the sliding median results when borders are set to None.

        Args:
            data_2d (tuple[np.ndarray, np.ndarray]): the kernel and data to test for.
        """

        kernel, data = data_2d
        expected = self._known_median_none()

        result = SlidingMedian(
            data=data,
            kernel=kernel,
            borders=None,
        ).median

        np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-5)

    def test_constant_std_sliding(
            self,
            data_2d: tuple[np.ndarray, np.ndarray],
        ) -> None:
        """
        Tests the sliding median results when borders are set to 'constant'.

        Args:
            data_2d (tuple[np.ndarray, np.ndarray]): the kernel and data to test for.
        """

        kernel, data = data_2d
        expected = self._known_std_constant()

        result = SlidingStandardDeviation(
            data=data,
            kernel=kernel,
            borders='constant',
        ).standard_deviation

        np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-5)

    def test_none_std_sliding(
            self,
            data_2d: tuple[np.ndarray, np.ndarray],
        ) -> None:
        """
        Tests the SlidingStandardDeviation class when borders are set to None.

        Args:
            data_2d (tuple[np.ndarray, np.ndarray]): the kernel and data to test for.
        """

        kernel, data = data_2d
        expected = self._known_std_none()

        result = SlidingStandardDeviation(
            data=data,
            kernel=kernel,
            borders=None,
        ).standard_deviation

        np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-5)
