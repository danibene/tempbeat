from neurokit2 import data, ecg_process

from tempbeat.evaluation.heartbeat_evaluation import get_bpm_mae_from_rri
from tempbeat.preprocessing.preprocessing_heartbeat import peak_time_to_rri
from tempbeat.preprocessing.preprocessing_utils import samp_to_timestamp


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
