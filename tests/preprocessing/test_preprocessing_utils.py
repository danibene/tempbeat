import numpy as np

from tempbeat.preprocessing.preprocessing_utils import (
    check_uniform_sig_time,
    peak_time_to_rri,
    rri_to_peak_time,
    samp_to_timestamp,
    sampling_rate_to_sig_time,
    sig_time_to_sampling_rate,
    timestamp_to_samp,
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


class TestSampToTimestamp:
    """
    Test cases for the samp_to_timestamp function.
    """

    @staticmethod
    def test_samp_to_timestamp_basic() -> None:
        """
        Test samp_to_timestamp with a basic example.

        The function should correctly convert sample indices to timestamps.
        """
        sampling_rate = 1000
        samp = np.array([100, 500, 1000])
        result = samp_to_timestamp(samp, sampling_rate=sampling_rate)
        # Subtract 1/sampling_rate to account for the fact that the first sample is at time 0.
        expected = np.array([0.1, 0.5, 1.0]) - 1 / sampling_rate
        np.testing.assert_array_equal(result, expected)

    @staticmethod
    def test_samp_to_timestamp_with_sig_time() -> None:
        """
        Test samp_to_timestamp with providing sig_time.

        The function should correctly convert sample indices to timestamps with sig_time.
        """
        sig_time = np.array([0.0, 0.1, 0.2, 0.3, 0.4])
        samp = np.array([0, 2, 4])
        result = samp_to_timestamp(samp, sig_time=sig_time)
        expected = np.array([0.0, 0.2, 0.4])
        np.testing.assert_array_equal(result, expected)

    @staticmethod
    def test_samp_to_timestamp_with_index_higher_than_sig_time_len() -> None:
        """
        Test samp_to_timestamp with providing sig_time and sample index higher than sig_time length.

        The function should take the last timestamp in sig_time as the timestamp for the sample index
        that is higher than sig_time length.
        """
        sig_time = np.array([0.0, 0.1, 0.2, 0.3, 0.4])
        samp = np.array([0, 2, 4, 6])
        result = samp_to_timestamp(samp, sig_time=sig_time)
        expected = np.array([0.0, 0.2, 0.4, 0.4])
        np.testing.assert_array_equal(result, expected)


class TestTimestampToSamp:
    """
    Test cases for the timestamp_to_samp function.
    """

    @staticmethod
    def test_timestamp_to_samp_basic() -> None:
        """
        Test timestamp_to_samp with a basic example.

        The function should correctly convert timestamps to sample indices.
        """
        sampling_rate = 1000
        timestamp = np.array([0.1, 0.5, 1.0])
        result = timestamp_to_samp(timestamp, sampling_rate=sampling_rate)
        expected = np.array([101, 501, 1001])
        np.testing.assert_array_equal(result, expected)

        sampling_rate = 100
        timestamp = np.array([0.1, 0.5, 1.0])
        result = timestamp_to_samp(timestamp, sampling_rate=sampling_rate)
        expected = np.array([11, 51, 101])
        np.testing.assert_array_equal(result, expected)

    @staticmethod
    def test_timestamp_to_samp_with_sig_time() -> None:
        """
        Test timestamp_to_samp with providing sig_time.

        The function should correctly convert timestamps to sample indices with sig_time.
        """
        sig_time = np.array([0.0, 0.1, 0.2, 0.3, 0.4])
        timestamp = np.array([0.0, 0.2, 0.4])
        result = timestamp_to_samp(timestamp, sig_time=sig_time)
        expected = np.array([0, 2, 4])
        np.testing.assert_array_equal(result, expected)


class TestCheckUniformSigTime:
    """
    Test cases for the check_uniform_sig_time function.
    """

    @staticmethod
    def test_check_uniform_sig_time_uniform() -> None:
        """
        Test check_uniform_sig_time with uniform timepoints.

        The function should return True when the difference between timepoints is uniform.
        """
        sig_time = np.array([0, 1, 2, 3, 4])
        result = check_uniform_sig_time(sig_time)
        assert result is True

    @staticmethod
    def test_check_uniform_sig_time_non_uniform() -> None:
        """
        Test check_uniform_sig_time with non-uniform timepoints.

        The function should return False when the difference between timepoints is not uniform.
        """
        sig_time = np.array([0, 0.5, 0.99, 1.5])
        result = check_uniform_sig_time(sig_time)
        assert result is False


class TestSigTimeToSamplingRate:
    """
    Test cases for the sig_time_to_sampling_rate function.
    """

    @staticmethod
    def test_sig_time_to_sampling_rate_median() -> None:
        """
        Test sig_time_to_sampling_rate with the median method.

        The function should correctly calculate the sampling rate using the median method.
        """
        sig_time = np.array([0, 1, 2, 3, 4])
        result = sig_time_to_sampling_rate(sig_time)
        expected = 1
        assert result == expected

    @staticmethod
    def test_sig_time_to_sampling_rate_mode() -> None:
        """
        Test sig_time_to_sampling_rate with the mode method.

        The function should correctly calculate the sampling rate using the mode method.
        """
        sig_time = np.array([0, 0.5, 1.0, 1.5])
        result = sig_time_to_sampling_rate(sig_time, method="mode")
        expected = 2
        assert result == expected

    @staticmethod
    def test_sig_time_to_sampling_rate_check_uniform(recwarn) -> None:
        """
        Test sig_time_to_sampling_rate with check_uniform=True.

        The function should correctly calculate the sampling rate with check_uniform=True.
        """
        sig_time = np.array([0, 0.5, 1.0, 1.3, 1.5])
        result = sig_time_to_sampling_rate(sig_time, check_uniform=True)
        expected = 2
        assert result == expected
        # assert that warning is issued
        assert len(recwarn) == 1

    @staticmethod
    def test_sig_time_to_sampling_rate_check_uniform_false(recwarn) -> None:
        """
        Test sig_time_to_sampling_rate with check_uniform=False.

        The function should correctly calculate the sampling rate with check_uniform=False.
        """
        sig_time = np.array([0, 0.5, 1.0, 1.3, 1.5])
        result = sig_time_to_sampling_rate(sig_time, check_uniform=False)
        expected = 2
        assert result == expected
        # assert that warning is not issued
        assert len(recwarn) == 0

    @staticmethod
    def test_sig_time_to_sampling_rate_check_uniform_decimals2(recwarn) -> None:
        """
        Test sig_time_to_sampling_rate with check_uniform=True and decimals=2.

        The function should correctly calculate the sampling rate with check_uniform=True and decimals=1.
        """
        sig_time = np.array([0, 0.5, 1.0, 1.5001, 2.0])
        result = sig_time_to_sampling_rate(sig_time, check_uniform=True, decimals=2)
        expected = 2
        assert result == expected
        # assert that warning is not issued
        assert len(recwarn) == 0

    @staticmethod
    def test_sig_time_to_sampling_rate_check_uniform_decimals5(recwarn) -> None:
        """
        Test sig_time_to_sampling_rate with check_uniform=True and decimals=5.

        The function should correctly calculate the sampling rate with check_uniform=True and decimals=5.
        """
        sig_time = np.array([0, 0.5, 1.0, 1.5001, 2.0])
        result = sig_time_to_sampling_rate(sig_time, check_uniform=True, decimals=5)
        expected = 2
        assert result == expected
        # assert that warning is issued
        assert len(recwarn) == 1


class TestSamplingRateToSigTime:
    """
    Test cases for the sampling_rate_to_sig_time function.
    """

    @staticmethod
    def test_sampling_rate_to_sig_time_basic() -> None:
        """
        Test sampling_rate_to_sig_time with a basic example.

        The function should correctly generate an array of timestamps corresponding to each sample.
        """
        sig = np.array([1, 2, 3])
        result = sampling_rate_to_sig_time(sig, sampling_rate=1000, start_time=0)
        expected = np.array([0.0, 0.001, 0.002])
        np.testing.assert_array_equal(result, expected)

    @staticmethod
    def test_sampling_rate_to_sig_time_start_time1() -> None:
        """
        Test sampling_rate_to_sig_time with a start_time of 1.

        The function should correctly generate an array of timestamps corresponding to each sample with start_time.
        """
        sig = np.array([1, 2, 3])
        result = sampling_rate_to_sig_time(sig, sampling_rate=1000, start_time=1)
        expected = np.array([1.0, 1.001, 1.002])
        np.testing.assert_array_equal(result, expected)
