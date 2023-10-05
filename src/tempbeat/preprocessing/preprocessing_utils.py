from typing import List, Optional, Tuple, Union
from warnings import warn

import numpy as np
import scipy.stats
from neurokit2.signal import signal_interpolate


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


def norm_corr(a: np.ndarray, b: np.ndarray, maxlags: int = 0) -> np.ndarray:
    """
    Calculate normalized cross-correlation between two 1-dimensional arrays.

    Parameters
    ----------
    a : np.ndarray
        First array.
    b : np.ndarray
        Second array.
    maxlags : int, optional
        Maximum lag to calculate. Default is 0.

    Returns
    -------
    np.ndarray
        Array containing the normalized cross-correlation.

    References
    ----------
    https://stackoverflow.com/questions/53436231/normalized-cross-correlation-in-python
    """
    Nx = len(a)

    if Nx != len(b):
        raise ValueError("a and b must be equal length")

    if maxlags is None:
        maxlags = Nx - 1

    if maxlags >= Nx or maxlags < 0:
        raise ValueError("maxlags must be None or strictly positive < %d" % Nx)

    a = (a - np.mean(a)) / (np.std(a) * len(a))
    b = (b - np.mean(b)) / (np.std(b))
    c = np.correlate(a, b, "full")
    c = c[Nx - 1 - maxlags : Nx + maxlags]

    return c


def a_moving_average(y: np.ndarray, N: int = 5) -> np.ndarray:
    """
    Calculate the moving average of a 1-dimensional array.

    Parameters
    ----------
    y : np.ndarray
        Input array.
    N : int, optional
        Number of points used for the moving average window. Default is 5.

    Returns
    -------
    np.ndarray
        Smoothed array after applying the moving average.
    """
    y_padded = np.pad(y, (N // 2, N - 1 - N // 2), mode="edge")
    y_smooth = np.convolve(y_padded, np.ones((N,)) / N, mode="valid")
    return y_smooth


def roll_func(
    x: np.ndarray, window: int, func: callable, func_args: dict = {}
) -> np.ndarray:
    """
    Apply a rolling function to the input array.

    Parameters
    ----------
    x : np.ndarray
        Input array.
    window : int
        Size of the rolling window.
    func : callable
        Function to apply to each window.
    func_args : dict, optional
        Additional arguments to pass to the function.

    Returns
    -------
    np.ndarray
        Array resulting from applying the rolling function.
    """
    roll_x = np.array(
        [func(x[i : i + window], **func_args) for i in range(len(x) - window)]
    )
    return roll_x


def scale_and_clip_to_max_one(
    x: np.ndarray,
    min_value: float = 0,
    replace_min_value: float = 0,
    max_value: float = np.inf,
    replace_max_value: float = None,
    div_by_given_max: bool = True,
) -> np.ndarray:
    """
    Scale and clip values in an array to be between 0 and 1.

    Parameters
    ----------
    x : np.ndarray
        Input array.
    min_value : float, optional
        Minimum value to clip, by default 0.
    replace_min_value : float, optional
        Value to replace elements below `min_value`, by default 0.
    max_value : float, optional
        Maximum value to clip, by default np.inf.
    replace_max_value : float, optional
        Value to replace elements above `max_value`, by default None.
        If None, `max_value` is used.
    div_by_given_max : bool, optional
        If True, divide the array by `max_value`, by default True.
        If False, divide by the maximum value in the array.

    Returns
    -------
    np.ndarray
        Scaled and clipped array.
    """
    if replace_max_value is None:
        replace_max_value = max_value
    x[x < min_value] = replace_min_value
    x[x > max_value] = replace_max_value
    if div_by_given_max:
        return x / max_value
    else:
        return x / np.nanmax(x)


def interpolate_to_same_x(
    a_x: np.ndarray,
    a_y: np.ndarray,
    b_x: np.ndarray,
    b_y: np.ndarray,
    interpolate_method: str = "linear",
    interpolation_rate: int = 2,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Interpolate two arrays to have the same x values.

    Parameters
    ----------
    a_x : np.ndarray
        x values for the first array.
    a_y : np.ndarray
        y values for the first array.
    b_x : np.ndarray
        x values for the second array.
    b_y : np.ndarray
        y values for the second array.
    interpolate_method : str, optional
        Interpolation method, by default "linear".
    interpolation_rate : int, optional
        Interpolation rate, by default 2.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        Tuple containing the x values and interpolated y values for the first and second arrays.
    """
    min_x = np.min([np.min(a_x), np.min(b_x)])
    max_x = np.max([np.max(a_x), np.max(b_x)])
    new_x = np.arange(min_x, max_x, 1 / interpolation_rate)
    a_y_interpolated = signal_interpolate(
        x_values=a_x, y_values=a_y, x_new=new_x, method=interpolate_method
    )
    b_y_interpolated = signal_interpolate(
        x_values=b_x, y_values=b_y, x_new=new_x, method=interpolate_method
    )
    return new_x, a_y_interpolated, b_y_interpolated
