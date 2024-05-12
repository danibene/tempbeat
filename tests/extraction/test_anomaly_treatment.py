import numpy as np

from tempbeat.extraction.anomaly_treatment import _get_added_kept_peaks


class TestGetAddedKeptPeaks:
    """
    Test cases for the _get_added_kept_peaks function.
    """

    @staticmethod
    def test_get_added_kept_peaks_basic() -> None:
        """
        Test _get_added_kept_peaks with a basic example.

        The function should correctly return the peaks that were added and kept.
        """
        fixed_peaks = np.array([1, 2, 3, 4, 5])
        kept_peaks = np.array([2, 4])
        added_peaks = np.array([1, 3, 5])
        added_peaks_result, kept_peaks_result = _get_added_kept_peaks(
            fixed_peaks, kept_peaks
        )
        np.testing.assert_array_equal(added_peaks_result, added_peaks)
        np.testing.assert_array_equal(kept_peaks_result, kept_peaks)
