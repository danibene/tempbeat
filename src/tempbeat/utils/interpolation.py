from typing import Tuple, Union

import numpy as np
from neurokit2.signal import signal_interpolate
from scipy import interpolate


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


def interpl_intervals_preserve_nans(
    x_old: np.ndarray, y_old: np.ndarray, x_new: np.ndarray
) -> np.ndarray:
    """
    Interpolate intervals (e.g. RRIs), preserving NaN values.

    Parameters
    ----------
    x_old : np.ndarray
        Old x values, each being a timestamp in seconds.
    y_old : np.ndarray
        Old y values, each being an interval (e.g. RRI) in milliseconds. Should be the same length as x_old.
    x_new : np.ndarray
        New x values for interpolation.

    Returns
    -------
    np.ndarray
        Interpolated y values.
    """
    x_old = x_old[np.isfinite(y_old)]
    y_old = y_old[np.isfinite(y_old)]
    y_new_nan = np.ones(x_new.size).astype(bool)
    step = np.median(np.diff(x_new))
    # Identify valid intervals using interval size and corresponding timestamps
    for i in range(len(x_old)):
        if i != 0:
            if np.abs((x_old[i] - (y_old[i] / 1000)) - x_old[i - 1]) < step:
                y_new_nan[
                    (x_new >= x_old[i] - (y_old[i] / 1000)) & (x_new <= x_old[i])
                ] = False
        y_new_nan[np.argmin(np.abs(x_new - x_old[i]))] = False
    f = interpolate.interp1d(x_old, y_old, kind="linear", fill_value="extrapolate")
    y_new = f(x_new)
    y_new[y_new_nan] = np.nan
    return y_new
