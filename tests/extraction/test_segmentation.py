import numpy as np
from neurokit2 import ecg_process, ecg_simulate

from tempbeat.extraction.segmentation import find_local_hb_peaks, get_local_hb_sig
from tempbeat.utils.timestamp import samp_to_timestamp


class TestFindLocalHbPeaks:
    """
    Test cases for the find_local_hb_peaks function.
    """

    @staticmethod
    def test_find_local_hb_peaks_basic() -> None:
        """
        Test find_local_hb_peaks with a basic example.

        The function should correct the peak times for each heartbeat segment.
        """
        sampling_rate = 1000
        sig = ecg_simulate(duration=10, sampling_rate=1000)
        _, rpeaks = ecg_process(sig, sampling_rate=sampling_rate)
        peaks = rpeaks["ECG_R_Peaks"]
        peak_time = samp_to_timestamp(peaks, sampling_rate=sampling_rate)
        assert isinstance(peak_time, np.ndarray)
        rng = np.random.default_rng(42)
        noisy_peak_time = peak_time + rng.uniform(-0.1, 0.1, size=peak_time.shape)
        time_before_peak = 0.2
        time_after_peak = 0.2
        local_hb_peaks_time = find_local_hb_peaks(
            noisy_peak_time,
            sig,
            sampling_rate=sampling_rate,
            time_before_peak=time_before_peak,
            time_after_peak=time_after_peak,
        )
        mean_diff_noisy_peak_time_original = np.mean(
            np.abs(noisy_peak_time - peak_time)
        )
        mean_diff_local_hb_peaks_time_original = np.mean(
            np.abs(local_hb_peaks_time - peak_time)
        )
        assert (
            mean_diff_local_hb_peaks_time_original < mean_diff_noisy_peak_time_original
        )


class TestGetLocalHbSig:
    """
    Test cases for the get_local_hb_sig function.
    """

    @staticmethod
    def test_get_local_hb_sig_basic() -> None:
        """
        Test get_local_hb_sig with a basic example.

        The function should correctly return the local heartbeat signal.
        """
        peak = 1.5
        sig = np.array([1, 2, 3, 4, 3, 2, 1])
        sig_time = np.array([0, 0.5, 1, 1.5, 2, 2.5, 3])
        time_before_peak = 0.6
        time_after_peak = 0.6
        hb_sig, hb_sig_time = get_local_hb_sig(
            peak,
            sig,
            sig_time=sig_time,
            time_before_peak=time_before_peak,
            time_after_peak=time_after_peak,
        )
        np.testing.assert_array_equal(hb_sig, np.array([3, 4, 3]))
        np.testing.assert_array_equal(hb_sig_time, np.array([1, 1.5, 2]))
