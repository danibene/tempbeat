from neurokit2 import data, ecg_process, signal_distort

from tempbeat.evaluation.compare_bpm import get_bpm_mae_from_rri
from tempbeat.extraction.interval_conversion import peak_time_to_rri
from tempbeat.utils.timestamp import samp_to_timestamp


class TestGetBPMMaeFromRRI:
    """
    Test cases for the get_bpm_mae_from_rri function.
    """

    @staticmethod
    def test_get_bpm_mae_from_rri_with_same_signal() -> None:
        """
        Test get_bpm_mae_from_rri with two copies of the same signal.

        The function should return a mean absolute error of 0.
        """
        sampling_rate = 100
        ecg_data = data("bio_resting_5min_100hz")
        clean_ecg = ecg_data["ECG"].values
        _, clean_rpeaks = ecg_process(clean_ecg, sampling_rate=sampling_rate)
        clean_peak_time = samp_to_timestamp(
            clean_rpeaks["ECG_R_Peaks"], sampling_rate=sampling_rate
        )
        rri_clean, rri_time_clean = peak_time_to_rri(clean_peak_time)
        mae = get_bpm_mae_from_rri(
            rri_a=rri_clean,
            rri_b=rri_clean,
            rri_time_a=rri_time_clean,
            rri_time_b=rri_time_clean,
        )
        assert mae == 0

    @staticmethod
    def test_get_bpm_mae_from_rri_with_distorted_signal() -> None:
        """
        Test get_bpm_mae_from_rri with a distorted signal.

        The function should return a mean absolute error greater than 0.
        """
        random_state = 42
        sampling_rate = 100
        ecg_data = data("bio_resting_5min_100hz")
        clean_ecg = ecg_data["ECG"].values
        duration = len(clean_ecg) / sampling_rate
        _, clean_rpeaks = ecg_process(clean_ecg, sampling_rate=sampling_rate)
        clean_peak_time = samp_to_timestamp(
            clean_rpeaks["ECG_R_Peaks"], sampling_rate=sampling_rate
        )
        rri_clean, rri_time_clean = peak_time_to_rri(clean_peak_time)

        distorted_ecg = signal_distort(
            clean_ecg,
            sampling_rate=sampling_rate,
            noise_amplitude=0.5,
            artifacts_amplitude=1,
            artifacts_number=int(duration / 10),
            artifacts_frequency=2,
            random_state=random_state,
        )
        _, distorted_rpeaks = ecg_process(distorted_ecg, sampling_rate=sampling_rate)
        distorted_peak_time = samp_to_timestamp(
            distorted_rpeaks["ECG_R_Peaks"], sampling_rate=sampling_rate
        )
        rri_distorted, rri_time_distored = peak_time_to_rri(distorted_peak_time)
        mae_clean_distorted = get_bpm_mae_from_rri(
            rri_a=rri_clean,
            rri_b=rri_distorted,
            rri_time_a=rri_time_clean,
            rri_time_b=rri_time_distored,
        )
        # Confirm that the R-R intervals extracted from distorted signal are different from the clean signal
        assert mae_clean_distorted > 0
