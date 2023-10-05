import numpy as np
from neurokit2 import ecg_process, ecg_simulate

from tempbeat.preprocessing.preprocessing_heartbeat import (
    find_local_hb_peaks,
    get_local_hb_sig,
    interpl_intervals_preserve_nans,
    peak_time_to_rri,
    rri_to_peak_time,
)
from tempbeat.preprocessing.preprocessing_utils import samp_to_timestamp


class TestPeakTimeToRRI:
    """
    Test cases for the peak_time_to_rri function.
    """

    @staticmethod
    def test_peak_time_to_rri_basic() -> None:
        """
        Test peak_time_to_rri with a basic example.

        The function should correctly convert peak times to R-R intervals without a minimum limit.
        """
        peak_time = np.array([1, 1.75, 2.25, 3.5, 4.5])
        result = peak_time_to_rri(peak_time)
        expected = (np.array([750, 500, 1250, 1000]), np.array([1.75, 2.25, 3.5, 4.5]))
        np.testing.assert_array_equal(result, expected)

    @staticmethod
    def test_peak_time_to_rri_min_max_rri() -> None:
        """
        Test peak_time_to_rri with minimum and maximum R-R intervals specified.

        The function should correctly convert peak times to R-R intervals.
        """
        peak_time = np.array([1, 1.75, 2.25, 3.5, 4.5])
        min_rri = 500  # Minimum R-R interval in milliseconds
        max_rri = 1200  # Maximum R-R interval in milliseconds
        result = peak_time_to_rri(peak_time, min_rri, max_rri)
        expected = (np.array([750, 500, 1000]), np.array([1.75, 2.25, 4.5]))
        np.testing.assert_array_equal(result, expected)

    @staticmethod
    def test_peak_time_to_rri_empty() -> None:
        """
        Test peak_time_to_rri with an empty array.

        The function should return an empty array.
        """
        peak_time = np.array([])
        result = peak_time_to_rri(peak_time)
        expected = (np.array([]), np.array([]))
        np.testing.assert_array_equal(result, expected)


class TestRRIToPeakTime:
    """
    Test cases for the rri_to_peak_time function.
    """

    @staticmethod
    def test_rri_to_peak_time_basic() -> None:
        """
        Test rri_to_peak_time with a basic example.

        The function should correctly convert R-R intervals to peak times.
        """
        rri = np.array([750, 500, 1000])
        rri_time = np.array([1.75, 2.25, 3.25])
        result = rri_to_peak_time(rri, rri_time)
        expected = np.array([1, 1.75, 2.25, 3.25])
        np.testing.assert_array_equal(result, expected)

    @staticmethod
    def test_rri_to_peak_time_empty() -> None:
        """
        Test rri_to_peak_time with an empty array.

        The function should return an empty array.
        """
        rri = np.array([])
        rri_time = np.array([])
        result = rri_to_peak_time(rri, rri_time)
        expected = np.array([])
        np.testing.assert_array_equal(result, expected)

    @staticmethod
    def test_rri_to_peak_time_negative_rri() -> None:
        """
        Test rri_to_peak_time with a negative R-R interval.

        The function should return an empty array.
        """
        rri = np.array([-500])
        rri_time = np.array([1])
        result = rri_to_peak_time(rri, rri_time)
        expected = np.array([])
        np.testing.assert_array_equal(result, expected)

    @staticmethod
    def test_rri_to_peak_time_to_rri() -> None:
        """
        Test rri_to_peak_time and peak_time_to_rri together.

        The function should correctly convert R-R intervals to peak times and back to R-R intervals.
        """
        rri = np.array([750, 500, 1000])
        rri_time = np.array([1.75, 2.25, 3.25])
        peak_time = rri_to_peak_time(rri, rri_time)
        result = peak_time_to_rri(peak_time)
        expected = (np.array([750, 500, 1000]), np.array([1.75, 2.25, 3.25]))
        np.testing.assert_array_equal(result, expected)


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


class TestInterplIntervalsPreserveNans:
    @staticmethod
    def test_interpl_intervals_preserve_nans_linear_interpolation() -> None:
        """
        Test linear interpolation with NaN preservation.
        """
        rri = np.array([500, 1000, np.nan, 750])
        rri_time = np.array([1, 2, 3, 4])
        x_new = np.array([1, 1.5, 2, 2.5, 3, 3.5, 4])
        result = interpl_intervals_preserve_nans(rri_time, rri, x_new)
        expected = np.array([500, 750, 1000, np.nan, np.nan, np.nan, 750])
        np.testing.assert_array_equal(result, expected)
