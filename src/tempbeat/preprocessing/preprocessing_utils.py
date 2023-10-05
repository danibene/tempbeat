from typing import Optional, Tuple, Union

import numpy as np
import scipy.stats
from neurokit2.signal import signal_interpolate

from .timestamps import sig_time_to_sampling_rate


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


def drop_missing(
    sig: np.ndarray,
    sig_time: Optional[np.ndarray] = None,
    missing_value: float = np.nan,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Drop missing values from a signal.

    This function drops missing values from a signal. If a signal time array is provided, the
    function also drops the corresponding timestamps.

    Parameters
    ----------
    sig : np.ndarray
        Input signal.
    sig_time : Optional[np.ndarray], optional
        Array of timestamps corresponding to each sample. If not provided, timestamps
        are calculated based on the sampling rate.
    missing_value : float, optional
        Value to be considered missing. Default is np.nan.

    Returns
    -------
    Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]
        If a signal time array is provided, a tuple containing the signal and signal time arrays
        with missing values dropped. Otherwise, the signal array with missing values dropped.
    """
    if np.isnan(missing_value):
        not_missing = np.invert(np.isnan(sig))
    else:
        not_missing = np.where(sig == missing_value)
    sig = sig[not_missing]
    if sig_time is not None:
        sig_time = sig_time[not_missing]
        return sig, sig_time
    return sig
