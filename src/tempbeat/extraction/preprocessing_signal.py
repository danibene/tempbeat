from typing import Optional, Tuple, Union

import numpy as np


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
