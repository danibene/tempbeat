import numpy as np

from tempbeat.utils.interpolation import interpl_intervals_preserve_nans


class TestInterplIntervalsPreserveNans:
    """
    Test cases for the interpl_intervals_preserve_nans function.
    """

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
