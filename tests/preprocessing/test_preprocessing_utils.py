import numpy as np

from tempbeat.preprocessing.preprocessing_utils import (
    peak_time_to_rri,
    rri_to_peak_time,
)


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
