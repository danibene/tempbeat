import numpy as np
import pytest

from tempbeat.extraction.correlation import norm_corr


class TestNormCorr:
    """
    Test cases for the norm_corr function.
    """

    @pytest.fixture
    def sample_data(self) -> tuple:
        arr_a = np.array([1, 2, 3, 4, 5])
        arr_b = np.array([2, 3, 4, 5, 6])
        return arr_a, arr_b

    def test_norm_corr_equal_length(self, sample_data: tuple) -> None:
        arr_a, arr_b = sample_data
        result = norm_corr(arr_a, arr_b)
        expected_result = 1
        np.testing.assert_allclose(result, expected_result, rtol=1e-6)

    def test_norm_corr_with_maxlags(self, sample_data: tuple) -> None:
        arr_a, arr_b = sample_data
        result = np.max(norm_corr(arr_a, arr_b, maxlags=2))
        expected_result = 1
        np.testing.assert_allclose(result, expected_result, rtol=1e-6)

    def test_norm_corr_unequal_length(self) -> None:
        arr_a = np.array([1, 2, 3, 4, 5])
        arr_c = np.array([1, 2, 3])
        with pytest.raises(ValueError):
            norm_corr(arr_a, arr_c)

    def test_norm_corr_invalid_maxlags(self, sample_data: tuple) -> None:
        arr_a, arr_b = sample_data
        with pytest.raises(ValueError):
            norm_corr(arr_a, arr_b, maxlags=10)
