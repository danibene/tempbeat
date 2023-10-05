import neurokit2 as nk
import numpy as np

from tempbeat.preprocessing.heartbeat_extract import temp_hb_extract
from tempbeat.preprocessing.preprocessing_utils import sampling_rate_to_sig_time


class TestTempHbExtract:
    """
    Test cases for the temp_hb_extract function.
    """

    @staticmethod
    def test_temp_hb_extract_regression() -> None:
        """
        Test temp_hb_extract with example ECG data.

        The function should extract the peak times such that they are within 0.01 seconds of the expected values.
        The expected values were obtained by running the function on the example ECG data with the same parameters.
        """
        sampling_rate = 100
        data = nk.data("bio_resting_5min_100hz")
        sig = data["ECG"]
        sig_time = sampling_rate_to_sig_time(sig, sampling_rate)
        peaks = temp_hb_extract(sig, sig_time=sig_time, sampling_rate=sampling_rate)
        np.testing.assert_allclose(
            peaks[:20],
            np.array(
                [
                    1.66,
                    2.27,
                    2.86,
                    3.47,
                    4.11,
                    4.77,
                    5.44,
                    6.13,
                    6.85,
                    7.49,
                    8.13,
                    8.79,
                    9.47,
                    10.12,
                    10.75,
                    11.4,
                    12.08,
                    12.77,
                    13.41,
                    14.04,
                ]
            ),
            rtol=0.01,
        )
