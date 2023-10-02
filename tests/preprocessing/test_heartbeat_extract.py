import numpy as np
from neurokit2 import ecg_simulate

from tempbeat.preprocessing.heartbeat_extract import temp_hb_extract
from tempbeat.preprocessing.preprocessing_utils import sampling_rate_to_sig_time


class TestTempHbExtract:
    """
    Test cases for the temp_hb_extract function.
    """

    @staticmethod
    def test_temp_hb_extract_regression() -> None:
        """
        Test temp_hb_extract with simulated ECG data.

        The function should extract the peak times such that they are within 0.01 seconds of the expected values.
        The expected values were obtained by running the function on the simulated ECG data with the same parameters.
        """
        sampling_rate = 1000
        random_state = 42
        random_state_distort = 43
        sig = ecg_simulate(
            duration=20,
            sampling_rate=sampling_rate,
            random_state=random_state,
            random_state_distort=random_state_distort,
        )
        sig_time = sampling_rate_to_sig_time(sig, sampling_rate)
        peaks = temp_hb_extract(sig, sig_time=sig_time, sampling_rate=sampling_rate)
        np.testing.assert_allclose(
            peaks,
            np.array(
                [
                    0.85,
                    1.72,
                    2.59,
                    3.44,
                    4.3,
                    5.18,
                    6.31,
                    7.45,
                    8.58,
                    9.72,
                    10.85,
                    11.99,
                    12.87,
                    13.75,
                    14.81,
                    15.88,
                    16.95,
                    18.02,
                    18.87,
                ]
            ),
            rtol=0.01,
        )
