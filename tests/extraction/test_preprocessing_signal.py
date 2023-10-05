from typing import Tuple

import numpy as np
import pytest
from neurokit2 import signal_power, signal_simulate

from tempbeat.extraction.correlation import norm_corr
from tempbeat.utils.interpolate import interpolate_nonuniform
from tempbeat.utils.resample import resample_nonuniform
from tempbeat.utils.timestamps import (
    check_uniform_sig_time,
    samp_to_timestamp,
    sampling_rate_to_sig_time,
    sig_time_to_sampling_rate,
    timestamp_to_samp,
)


class TestSampToTimestamp:
    """
    Test cases for the samp_to_timestamp function.
    """

    @staticmethod
    def test_samp_to_timestamp_basic() -> None:
        """
        Test samp_to_timestamp with a basic example.

        The function should correctly convert sample indices to timestamps.
        """
        sampling_rate = 1000
        samp = np.array([100, 500, 1000])
        result = samp_to_timestamp(samp, sampling_rate=sampling_rate)
        # Subtract 1/sampling_rate to account for the fact that the first sample is at time 0.
        expected = np.array([0.1, 0.5, 1.0]) - 1 / sampling_rate
        np.testing.assert_array_equal(result, expected)

    @staticmethod
    def test_samp_to_timestamp_with_sig_time() -> None:
        """
        Test samp_to_timestamp with providing sig_time.

        The function should correctly convert sample indices to timestamps with sig_time.
        """
        sig_time = np.array([0.0, 0.1, 0.2, 0.3, 0.4])
        samp = np.array([0, 2, 4])
        result = samp_to_timestamp(samp, sig_time=sig_time)
        expected = np.array([0.0, 0.2, 0.4])
        np.testing.assert_array_equal(result, expected)

    @staticmethod
    def test_samp_to_timestamp_with_index_higher_than_sig_time_len() -> None:
        """
        Test samp_to_timestamp with providing sig_time and sample index higher than sig_time length.

        The function should take the last timestamp in sig_time as the timestamp for the sample index
        that is higher than sig_time length.
        """
        sig_time = np.array([0.0, 0.1, 0.2, 0.3, 0.4])
        samp = np.array([0, 2, 4, 6])
        result = samp_to_timestamp(samp, sig_time=sig_time)
        expected = np.array([0.0, 0.2, 0.4, 0.4])
        np.testing.assert_array_equal(result, expected)


class TestTimestampToSamp:
    """
    Test cases for the timestamp_to_samp function.
    """

    @staticmethod
    def test_timestamp_to_samp_basic() -> None:
        """
        Test timestamp_to_samp with a basic example.

        The function should correctly convert timestamps to sample indices.
        """
        sampling_rate = 1000
        timestamp = np.array([0.1, 0.5, 1.0])
        result = timestamp_to_samp(timestamp, sampling_rate=sampling_rate)
        expected = np.array([101, 501, 1001])
        np.testing.assert_array_equal(result, expected)

        sampling_rate = 100
        timestamp = np.array([0.1, 0.5, 1.0])
        result = timestamp_to_samp(timestamp, sampling_rate=sampling_rate)
        expected = np.array([11, 51, 101])
        np.testing.assert_array_equal(result, expected)

    @staticmethod
    def test_timestamp_to_samp_with_sig_time() -> None:
        """
        Test timestamp_to_samp with providing sig_time.

        The function should correctly convert timestamps to sample indices with sig_time.
        """
        sig_time = np.array([0.0, 0.1, 0.2, 0.3, 0.4])
        timestamp = np.array([0.0, 0.2, 0.4])
        result = timestamp_to_samp(timestamp, sig_time=sig_time)
        expected = np.array([0, 2, 4])
        np.testing.assert_array_equal(result, expected)


class TestCheckUniformSigTime:
    """
    Test cases for the check_uniform_sig_time function.
    """

    @staticmethod
    def test_check_uniform_sig_time_uniform() -> None:
        """
        Test check_uniform_sig_time with uniform timepoints.

        The function should return True when the difference between timepoints is uniform.
        """
        sig_time = np.array([0, 1, 2, 3, 4])
        result = check_uniform_sig_time(sig_time)
        assert result is True

    @staticmethod
    def test_check_uniform_sig_time_non_uniform() -> None:
        """
        Test check_uniform_sig_time with non-uniform timepoints.

        The function should return False when the difference between timepoints is not uniform.
        """
        sig_time = np.array([0, 0.5, 0.99, 1.5])
        result = check_uniform_sig_time(sig_time)
        assert result is False


class TestSigTimeToSamplingRate:
    """
    Test cases for the sig_time_to_sampling_rate function.
    """

    @staticmethod
    def test_sig_time_to_sampling_rate_median() -> None:
        """
        Test sig_time_to_sampling_rate with the median method.

        The function should correctly calculate the sampling rate using the median method.
        """
        sig_time = np.array([0, 1, 2, 3, 4])
        result = sig_time_to_sampling_rate(sig_time)
        expected = 1
        assert result == expected

    @staticmethod
    def test_sig_time_to_sampling_rate_mode() -> None:
        """
        Test sig_time_to_sampling_rate with the mode method.

        The function should correctly calculate the sampling rate using the mode method.
        """
        sig_time = np.array([0, 0.5, 1.0, 1.5])
        result = sig_time_to_sampling_rate(sig_time, method="mode")
        expected = 2
        assert result == expected

    @staticmethod
    def test_sig_time_to_sampling_rate_check_uniform(recwarn) -> None:
        """
        Test sig_time_to_sampling_rate with check_uniform=True.

        The function should correctly calculate the sampling rate with check_uniform=True.
        """
        sig_time = np.array([0, 0.5, 1.0, 1.3, 1.5])
        result = sig_time_to_sampling_rate(sig_time, check_uniform=True)
        expected = 2
        assert result == expected
        # assert that warning is issued
        assert len(recwarn) == 1

    @staticmethod
    def test_sig_time_to_sampling_rate_check_uniform_false(recwarn) -> None:
        """
        Test sig_time_to_sampling_rate with check_uniform=False.

        The function should correctly calculate the sampling rate with check_uniform=False.
        """
        sig_time = np.array([0, 0.5, 1.0, 1.3, 1.5])
        result = sig_time_to_sampling_rate(sig_time, check_uniform=False)
        expected = 2
        assert result == expected
        # assert that warning is not issued
        assert len(recwarn) == 0

    @staticmethod
    def test_sig_time_to_sampling_rate_check_uniform_decimals2(recwarn) -> None:
        """
        Test sig_time_to_sampling_rate with check_uniform=True and decimals=2.

        The function should correctly calculate the sampling rate with check_uniform=True and decimals=1.
        """
        sig_time = np.array([0, 0.5, 1.0, 1.5001, 2.0])
        result = sig_time_to_sampling_rate(sig_time, check_uniform=True, decimals=2)
        expected = 2
        assert result == expected
        # assert that warning is not issued
        assert len(recwarn) == 0

    @staticmethod
    def test_sig_time_to_sampling_rate_check_uniform_decimals5(recwarn) -> None:
        """
        Test sig_time_to_sampling_rate with check_uniform=True and decimals=5.

        The function should correctly calculate the sampling rate with check_uniform=True and decimals=5.
        """
        sig_time = np.array([0, 0.5, 1.0, 1.5001, 2.0])
        result = sig_time_to_sampling_rate(sig_time, check_uniform=True, decimals=5)
        expected = 2
        assert result == expected
        # assert that warning is issued
        assert len(recwarn) == 1


class TestSamplingRateToSigTime:
    """
    Test cases for the sampling_rate_to_sig_time function.
    """

    @staticmethod
    def test_sampling_rate_to_sig_time_basic() -> None:
        """
        Test sampling_rate_to_sig_time with a basic example.

        The function should correctly generate an array of timestamps corresponding to each sample.
        """
        sig = np.array([1, 2, 3])
        result = sampling_rate_to_sig_time(sig, sampling_rate=1000, start_time=0)
        expected = np.array([0.0, 0.001, 0.002])
        np.testing.assert_array_equal(result, expected)

    @staticmethod
    def test_sampling_rate_to_sig_time_start_time1() -> None:
        """
        Test sampling_rate_to_sig_time with a start_time of 1.

        The function should correctly generate an array of timestamps corresponding to each sample with start_time.
        """
        sig = np.array([1, 2, 3])
        result = sampling_rate_to_sig_time(sig, sampling_rate=1000, start_time=1)
        expected = np.array([1.0, 1.001, 1.002])
        np.testing.assert_array_equal(result, expected)

    @staticmethod
    def test_sampling_rate_to_sig_time_diff() -> None:
        """
        Test that sampling_rate_to_sig_time generates a difference of 1/sampling_rate between each timestamp.

        The function should correctly generate an array of timestamps corresponding to each sample.
        """
        sampling_rate = 1000
        sig = np.array([1, 2, 3])
        sig_time = sampling_rate_to_sig_time(sig, sampling_rate=1000)
        assert np.all(np.diff(sig_time) == 1 / sampling_rate)


class TestInterpolateNonuniform:
    """
    Test cases for the interpolate_nonuniform function.
    """

    @staticmethod
    def test_interpolate_nonuniform_with_uniform_signal() -> None:
        """
        Test interpolate_nonuniform with a basic example.

        The function should correctly interpolate a signal with uniform timepoints.
        """
        sig = np.array([1, 2, 3])
        sig_time = np.array([0, 1, 2])
        new_sampling_rate = 2
        sig, sig_time = interpolate_nonuniform(
            sig, sig_time, sampling_rate=new_sampling_rate, method="linear"
        )
        expected = np.array([1, 1.5, 2, 2.5, 3])
        np.testing.assert_array_equal(sig, expected)

    @staticmethod
    def test_interpolate_nonuniform_with_nonuniform_signal() -> None:
        """
        Test interpolate_nonuniform with samples that are not uniformly spaced.

        The function should correctly interpolate a signal with non-uniform timepoints.
        """
        sig = np.array([1, 2, 3])
        sig_time = np.array([0, 1, 3])
        new_sampling_rate = 2
        sig, sig_time = interpolate_nonuniform(
            sig, sig_time, sampling_rate=new_sampling_rate, method="linear"
        )
        expected = np.array([1, 1.5, 2, 2.25, 2.5, 2.75, 3])
        np.testing.assert_array_equal(sig, expected)


class TestResampleNonuniform:
    """
    Test cases for the resample_nonuniform function.
    """

    @staticmethod
    def get_test_signal_uniform() -> Tuple[np.ndarray, np.ndarray]:
        """
        Get a test signal with uniform timepoints.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            A tuple containing the signal and the corresponding timepoints.
        """
        duration = 10
        sampling_rate = 1000
        # Generate signal with power at 15 Hz and 50 Hz
        amplitude15 = 1
        sig_freq15 = (
            signal_simulate(
                duration=duration, sampling_rate=sampling_rate, frequency=15
            )
            * amplitude15
        )
        amplitude50 = 0.5
        sig_freq50 = (
            signal_simulate(
                duration=duration, sampling_rate=sampling_rate, frequency=50
            )
            * amplitude50
        )
        sig = sig_freq15 + sig_freq50
        assert sig.shape[0] == duration * sampling_rate
        sig_time = sampling_rate_to_sig_time(sig, sampling_rate=sampling_rate)
        return sig, sig_time

    def get_test_signal_nonuniform(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get a test signal with non-uniform timepoints.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            A tuple containing the signal and the corresponding timepoints.
        """
        sig, sig_time = self.get_test_signal_uniform()
        random_seed = 42
        rng = np.random.default_rng(random_seed)
        indices_to_remove = rng.choice(np.arange(sig.shape[0]), size=100, replace=False)
        return np.delete(sig, indices_to_remove), np.delete(sig_time, indices_to_remove)

    def check_correctly_resampled(
        self,
        sig: np.ndarray,
        sig_time: np.ndarray,
        new_sig: np.ndarray,
        new_sig_time: np.ndarray,
        new_sampling_rate: int,
    ) -> None:
        """
        Check that the signal is correctly resampled.

        Parameters
        ----------
        sig : np.ndarray
            The original signal.
        sig_time : np.ndarray
            The original timepoints.
        new_sig : np.ndarray
            The resampled signal.
        new_sig_time : np.ndarray
            The resampled timepoints.
        new_sampling_rate : int
            The sampling rate of the resampled signal.
        """
        # Check that the signal is correctly resampled
        assert sig_time_to_sampling_rate(new_sig_time) == new_sampling_rate
        # Check that the power at 15 Hz and 50 Hz is preserved
        frequency_band = [(14, 16), (24, 26), (49, 51)]
        sampling_rate = sig_time_to_sampling_rate(sig_time)
        power_sig = signal_power(
            sig, frequency_band=frequency_band, sampling_rate=sampling_rate
        )
        power_new_sig = signal_power(
            new_sig, frequency_band=frequency_band, sampling_rate=new_sampling_rate
        )
        # Compare which frequency bands have the highest power
        bands_sorted_by_power_sig = np.argsort(power_sig)[::-1]
        bands_sorted_by_power_new_sig = np.argsort(power_new_sig)[::-1]
        # Check that the frequency bands with the highest power are the same
        assert np.array_equal(bands_sorted_by_power_sig, bands_sorted_by_power_new_sig)

    def test_resample_scipy_signal_uniform(self) -> None:
        """
        Test resample_nonuniform with scipy.

        The function should correctly resample a signal with uniform timepoints.
        """
        sig, sig_time = self.get_test_signal_uniform()
        # Resample signal to 500 Hz
        new_sampling_rate = 500
        new_sig, new_sig_time = resample_nonuniform(sig, sig_time, new_sampling_rate)
        self.check_correctly_resampled(
            sig, sig_time, new_sig, new_sig_time, new_sampling_rate=new_sampling_rate
        )

    def test_resample_scipy_signal_nonuniform(self) -> None:
        """
        Test resample_nonuniform with scipy.

        The function should correctly resample a signal with non-uniform timepoints.
        """
        sig, sig_time = self.get_test_signal_nonuniform()
        # Resample signal to 500 Hz
        new_sampling_rate = 500
        new_sig, new_sig_time = resample_nonuniform(sig, sig_time, new_sampling_rate)
        self.check_correctly_resampled(
            sig, sig_time, new_sig, new_sig_time, new_sampling_rate=new_sampling_rate
        )


class TestNormCorr:
    @pytest.fixture
    def sample_data(self):
        arr_a = np.array([1, 2, 3, 4, 5])
        arr_b = np.array([2, 3, 4, 5, 6])
        return arr_a, arr_b

    def test_norm_corr_equal_length(self, sample_data):
        arr_a, arr_b = sample_data
        result = norm_corr(arr_a, arr_b)
        expected_result = 1
        np.testing.assert_allclose(result, expected_result, rtol=1e-6)

    def test_norm_corr_with_maxlags(self, sample_data):
        arr_a, arr_b = sample_data
        result = np.max(norm_corr(arr_a, arr_b, maxlags=2))
        expected_result = 1
        np.testing.assert_allclose(result, expected_result, rtol=1e-6)

    def test_norm_corr_unequal_length(self):
        arr_a = np.array([1, 2, 3, 4, 5])
        arr_c = np.array([1, 2, 3])
        with pytest.raises(ValueError):
            norm_corr(arr_a, arr_c)

    def test_norm_corr_invalid_maxlags(self, sample_data):
        arr_a, arr_b = sample_data
        with pytest.raises(ValueError):
            norm_corr(arr_a, arr_b, maxlags=10)
