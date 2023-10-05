from typing import Tuple

import numpy as np
from neurokit2 import signal_power, signal_simulate

from tempbeat.utils.resampling import resample_nonuniform
from tempbeat.utils.timestamp import (
    sampling_rate_to_sig_time,
    sig_time_to_sampling_rate,
)


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
