from typing import List, Optional, Tuple, Union
from warnings import warn

import numpy as np
import scipy.stats
from neurokit2.hrv.intervals_utils import _intervals_successive
from neurokit2.signal import signal_interpolate


def peak_time_to_rri(
    peak_time: np.ndarray,
    min_rri: Optional[float] = None,
    max_rri: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert peak times to R-R intervals.

    This function takes an array of peak times and calculates the corresponding R-R intervals.
    It filters the R-R intervals based on optional minimum and maximum values.

    Parameters
    ----------
    peak_time : np.ndarray
        Array containing the times of detected peaks.
    min_rri : float, optional
        Minimum acceptable R-R interval. Default is None.
    max_rri : float, optional
        Maximum acceptable R-R interval. Default is None.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Tuple containing two arrays:
        - First array: R-R intervals that meet the criteria.
        - Second array: Corresponding times for the valid R-R intervals.
    """

    peak_time = np.sort(peak_time)
    rri = np.diff(peak_time) * 1000
    rri_time = peak_time[1:]

    if min_rri is None:
        min_rri = 0
    if max_rri is None:
        max_rri = np.inf

    keep = np.where((rri >= min_rri) & (rri <= max_rri))
    return rri[keep], rri_time[keep]


def rri_to_peak_time(rri: np.ndarray, rri_time: np.ndarray) -> np.ndarray:
    """
    Convert R-R intervals to peak times.

    This function takes arrays of R-R intervals and corresponding times and converts them to peak times.

    Parameters
    ----------
    rri : np.ndarray
        Array containing R-R intervals.
    rri_time : np.ndarray
        Array containing corresponding times for the R-R intervals.

    Returns
    -------
    np.ndarray
        Array containing peak times.
    """
    if len(rri_time) < 1:
        return rri_time

    keep_rri = np.where(rri > 0 & np.isfinite(rri))

    rri_time = rri_time[keep_rri]
    rri = rri[keep_rri]

    if len(rri_time) < 1:
        return rri_time

    non_successive_rri_ind = np.arange(1, len(rri_time))[
        np.invert(_intervals_successive(rri, rri_time, thresh_unequal=10))
    ]
    subtr_time_before_ind = np.concatenate((np.array([0]), non_successive_rri_ind))
    times_to_insert = (
        rri_time[subtr_time_before_ind] - rri[subtr_time_before_ind] / 1000
    )
    peak_time = np.sort(np.concatenate((rri_time, times_to_insert)))

    return peak_time


def samp_to_timestamp(
    samp: Union[int, np.ndarray],
    sampling_rate: int = 1000,
    sig_time: Optional[np.ndarray] = None,
) -> Union[float, np.ndarray]:
    """
    Convert sample indices to timestamps.

    This function takes sample indices and converts them to timestamps based on the provided
    sampling rate and optional signal time array. If a signal time array is provided, the
    function will ensure that the returned timestamps do not exceed the last timestamp in
    the array.

    Parameters
    ----------
    samp : Union[int, np.ndarray]
        Sample index or array of sample indices.
    sampling_rate : int, optional
        The sampling rate of the signal, in Hz. Default is 1000 Hz.
    sig_time : Optional[np.ndarray], optional
        Array of timestamps corresponding to each sample index. If not provided, timestamps
        are calculated based on the sampling rate.

    Returns
    -------
    Union[float, np.ndarray]
        Timestamp or array of timestamps corresponding to the input sample index or array.

    Warnings
    --------
    - If a sample index is less than 0, it is changed to 0.
    - If a sample index is greater than the last index, it is changed to the last index.
    """
    less_than_zero = np.where(samp < 0)
    if np.any(less_than_zero):
        warn(
            "Warning: the sample index is less than 0. Changing the sample index to 0."
        )
        samp[less_than_zero] = 0

    if sig_time is None:
        timestamp = samp / sampling_rate - 1 / sampling_rate
    else:
        bigger_than_last_index = np.where(samp >= len(sig_time))
        if np.any(bigger_than_last_index):
            warn(
                """Warning: the sample index is greater than the last index.
                Changing the sample index to the last index."""
            )
            samp[bigger_than_last_index] = len(sig_time) - 1
        timestamp = sig_time[samp]

    return timestamp


def timestamp_to_samp(
    timestamp: Union[float, np.ndarray],
    sampling_rate: int = 1000,
    sig_time: Optional[np.ndarray] = None,
    check_greater_than_last: bool = True,
) -> np.ndarray:
    """
    Convert timestamps to sample indices.

    This function takes timestamps and converts them to sample indices based on the provided
    sampling rate and optional signal time array.

    Parameters
    ----------
    timestamp : Union[float, np.ndarray]
        Timestamp or array of timestamps.
    sampling_rate : int, optional
        The sampling rate of the signal, in Hz. Default is 1000 Hz.
    sig_time : Optional[np.ndarray], optional
        Array of timestamps corresponding to each sample index. If not provided, timestamps
        are calculated based on the sampling rate.
    check_greater_than_last : bool, optional
        Whether to check if the calculated sample index is greater than the last index in
        `sig_time`. Default is True.

    Returns
    -------
    np.ndarray
        Array of sample indices corresponding to the input timestamp or array.

    Warnings
    --------
    - If a sample index is greater than the last index, it is changed to the last index.
    - If a sample index is less than 0, it is changed to 0.
    """
    timestamp = np.array(timestamp)
    if timestamp.size == 1:
        timestamp = np.array([timestamp])

    if sig_time is None:
        sig_time = [0]
        if check_greater_than_last:
            warn(
                """Warning: to check whether the sample is greater than the last sample index,
                 sig_time must be given."""
            )
            check_greater_than_last = False
        samp = np.array(
            np.round((timestamp - sig_time[0] + 1 / sampling_rate) * sampling_rate)
        ).astype(int)
    else:
        samp = np.array([np.argmin(np.abs(sig_time - t)) for t in timestamp]).astype(
            int
        )

    if check_greater_than_last:
        greater_than_len = np.where(samp > len(sig_time) - 1)
        if np.any(greater_than_len):
            warn(
                """Warning: the sample index is greater than the last sample index.
                 Changing the sample index to the last sample index."""
            )
            samp[greater_than_len] = len(sig_time) - 1

    less_than_zero = np.where(samp < 0)
    if np.any(less_than_zero):
        warn(
            "Warning: the sample index is less than 0. Changing the sample index to 0."
        )
        samp[less_than_zero] = 0

    return samp


def check_uniform_sig_time(
    sig_time: Union[np.ndarray, list], decimals: int = 6
) -> bool:
    """
    Check if the difference between timepoints in a signal time array is uniform.

    This function checks if the difference between consecutive timepoints in a signal time array
    is uniform up to a specified number of decimals.

    Parameters
    ----------
    sig_time : Union[np.ndarray, list]
        Array of timestamps corresponding to each sample.
    decimals : int, optional
        Number of decimal places to consider when checking uniformity. Default is 6.

    Returns
    -------
    bool
        True if the difference between timepoints is uniform, False otherwise.
    """
    return len(np.unique(np.round(np.diff(sig_time), decimals=decimals))) == 1


def sig_time_to_sampling_rate(
    sig_time: Union[np.ndarray, List[float]],
    method: str = "median",
    check_uniform: bool = True,
    decimals: int = 12,
) -> int:
    """
    Convert signal time array to sampling rate.

    This function calculates the sampling rate based on the provided signal time array using
    either the median or mode method.

    Parameters
    ----------
    sig_time : Union[np.ndarray, List[float]]
        Array of timestamps corresponding to each sample.
    method : str, optional
        Method to use for calculating the sampling rate. Either "median" (default) or "mode".
    check_uniform : bool, optional
        Whether to check if the difference between timepoints is uniform. Default is True.
    decimals : int, optional
        Number of decimal places to consider when checking uniformity. Default is 12.

    Returns
    -------
    int
        Calculated sampling rate.

    Warnings
    --------
    - If `check_uniform` is True and the difference between timepoints is not uniform, a warning
      is issued.

    Examples
    --------
    >>> sig_time_to_sampling_rate(np.array([0, 1, 2, 3, 4]))
    1

    >>> sig_time_to_sampling_rate(np.array([0, 0.5, 1.0, 1.5]), method="mode")
    2
    """
    if check_uniform:
        if not check_uniform_sig_time(sig_time, decimals=decimals):
            warn("Warning: the difference between timepoints is not uniform")

    if method == "mode":
        sampling_rate = int(1 / scipy.stats.mode(np.diff(sig_time)).mode)
    else:
        sampling_rate = int(1 / np.median(np.diff(sig_time)))

    return sampling_rate


def sampling_rate_to_sig_time(
    sig: Union[np.ndarray, List[float]],
    sampling_rate: int = 1000,
    start_time: float = 0,
) -> np.ndarray:
    """
    Convert sampling rate to signal time array.

    This function generates an array of timestamps corresponding to each sample based on the
    provided sampling rate and start time.

    Parameters
    ----------
    sig : Union[np.ndarray, List[float]]
        Input signal.
    sampling_rate : int, optional
        The sampling rate of the signal, in Hz. Default is 1000 Hz.
    start_time : float, optional
        Start time of the signal in seconds. Default is 0.

    Returns
    -------
    np.ndarray
        Array of timestamps corresponding to each sample.
    """
    sig_time = (np.arange(0, len(sig)) / sampling_rate) + start_time
    return sig_time


def resample_nonuniform(
    sig: Union[np.ndarray, list],
    sig_time: Union[np.ndarray, list],
    new_sampling_rate: int = 1000,
    interpolate_method: str = "linear",
    use_matlab: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Resample a non-uniformly sampled signal to a new sampling rate.

    Parameters
    ----------
    sig : Union[np.ndarray, list]
        Input signal.
    sig_time : Union[np.ndarray, list]
        Array of timestamps corresponding to each sample.
    new_sampling_rate : int, optional
        The desired new sampling rate. Default is 1000 Hz.
    interpolate_method : str, optional
        Interpolation method for non-uniformly sampled signal. Default is "linear".
    use_matlab : bool, optional
        Whether to use MATLAB for resampling. Default is False.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Tuple containing the resampled signal and corresponding timestamps.

    Notes
    -----
    If `use_matlab` is True, the function uses MATLAB for resampling. Otherwise, it uses
    scipy's resample function.
    """
    if use_matlab:
        return _resample_matlab(sig, sig_time, new_sampling_rate)
    else:
        return _resample_scipy(sig, sig_time, new_sampling_rate, interpolate_method)


def _resample_matlab(
    sig: Union[np.ndarray, list],
    sig_time: Union[np.ndarray, list],
    new_sampling_rate: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Resample using MATLAB.

    Parameters
    ----------
    sig : Union[np.ndarray, list]
        Input signal.
    sig_time : Union[np.ndarray, list]
        Array of timestamps corresponding to each sample.
    new_sampling_rate : int
        The desired new sampling rate.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Tuple containing the resampled signal and corresponding timestamps.
    """
    try:
        import matlab.engine
    except ImportError:
        raise ImportError(
            "To use MATLAB for resampling, you must have MATLAB installed and the "
            "matlab.engine package installed in Python."
        )

    eng = matlab.engine.start_matlab()
    eng.workspace["x"] = matlab.double(np.vstack(sig).astype(dtype="float64"))
    eng.workspace["tx"] = matlab.double(np.vstack(sig_time).astype(dtype="float64"))
    eng.workspace["fs"] = matlab.double(new_sampling_rate)
    y, ty = eng.eval("resample(x, tx, fs);", nargout=2)
    new_sig = np.hstack(np.asarray(y))
    new_sig_time = np.hstack(np.asarray(ty))
    eng.quit()
    return new_sig, new_sig_time


def _resample_scipy(
    sig: Union[np.ndarray, list],
    sig_time: Union[np.ndarray, list],
    new_sampling_rate: int,
    interpolate_method: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Resample using scipy.

    Parameters
    ----------
    sig : Union[np.ndarray, list]
        Input signal.
    sig_time : Union[np.ndarray, list]
        Array of timestamps corresponding to each sample.
    new_sampling_rate : int
        The desired new sampling rate.
    interpolate_method : str
        Interpolation method for non-uniformly sampled signal.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Tuple containing the resampled signal and corresponding timestamps.
    """
    sampling_rate_interpl = sig_time_to_sampling_rate(
        sig_time, method="median", check_uniform=False
    )
    sig_interpl, sig_time_interpl = interpolate_nonuniform(
        sig,
        sig_time,
        sampling_rate=sampling_rate_interpl,
        method=interpolate_method,
    )
    new_n_samples = int(
        np.round(len(sig_time_interpl) * (new_sampling_rate / sampling_rate_interpl))
    )
    new_sig, new_sig_time = scipy.signal.resample(
        sig_interpl, new_n_samples, t=sig_time_interpl
    )
    return new_sig, new_sig_time


def interpolate_nonuniform(
    sig: Union[np.ndarray, list],
    sig_time: Union[np.ndarray, list],
    sampling_rate: int,
    method: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Interpolate a non-uniformly sampled signal.

    Parameters
    ----------
    sig : Union[np.ndarray, list]
        Input signal.
    sig_time : Union[np.ndarray, list]
        Array of timestamps corresponding to each sample.
    sampling_rate : int
        The desired sampling rate.
    method : str
        Interpolation method.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Tuple containing the interpolated signal and corresponding timestamps.
    """
    start_sample_new = np.floor(sampling_rate * sig_time[0])
    end_sample_new = np.ceil(sampling_rate * sig_time[-1])
    new_sig_time = np.arange(start_sample_new, end_sample_new + 1) / sampling_rate
    new_sig = signal_interpolate(
        x_values=sig_time, y_values=sig, x_new=new_sig_time, method=method
    )
    return new_sig, new_sig_time


def get_local_hb_sig(
    peak: float,
    sig: np.ndarray,
    sig_time: Optional[np.ndarray] = None,
    sampling_rate: int = 1000,
    time_before_peak: float = 0.2,
    time_after_peak: float = 0.2,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get a local heartbeat signal around a peak.

    Parameters
    ----------
    peak : float
        The timestamp of the peak around which the local signal is extracted.
    sig : np.ndarray
        The signal.
    sig_time : np.ndarray, optional
        The timestamps corresponding to the signal samples. If None, it is assumed
        that the samples are uniformly spaced.
    sampling_rate : int, optional
        The sampling rate of the signal (in Hz, i.e., samples per second).
    time_before_peak : float, optional
        The duration of the signal to include before the peak (in seconds).
    time_after_peak : float, optional
        The duration of the signal to include after the peak (in seconds).

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        A tuple containing the local signal (`hb_sig`) and its corresponding timestamps (`hb_sig_time`).
    """
    if sig_time is None:
        # Assuming uniform spacing if timestamps are not provided
        sig_time = np.arange(0, len(sig)) / sampling_rate

    hb_sig_indices = np.where(
        (sig_time > peak - time_before_peak) & (sig_time < peak + time_after_peak)
    )
    hb_sig = sig[hb_sig_indices]
    hb_sig_time = sig_time[hb_sig_indices]

    return hb_sig, hb_sig_time
