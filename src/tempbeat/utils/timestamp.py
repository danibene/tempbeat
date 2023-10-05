from typing import List, Optional, Union
from warnings import warn

import numpy as np
import scipy


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
