from typing import Optional, Tuple

import neurokit2 as nk
import numpy as np
import scipy

from ..utils.misc_utils import argtop_k, get_func_kwargs
from ..utils.timestamps import (
    sampling_rate_to_sig_time,
    sig_time_to_sampling_rate,
    timestamp_to_samp,
)


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


def find_local_hb_peaks(
    peak_time: np.ndarray,
    sig: np.ndarray,
    sig_time: Optional[np.ndarray] = None,
    sampling_rate: int = 1000,
    check_height_outlier: bool = False,
    k_sample_ratio: float = 0.5,
    use_prominence: bool = False,
    **kwargs
) -> np.ndarray:
    """
    Find local peaks in a cardiac signal around specified peak times.

    Parameters
    ----------
    peak_time : np.ndarray
        Array of timestamps corresponding to the peaks in the cardiac signal.
    sig : np.ndarray
        The cardiac signal.
    sig_time : np.ndarray, optional
        Array of timestamps corresponding to the samples in the cardiac signal.
        If None, it is assumed that the samples are uniformly spaced.
    sampling_rate : int, optional
        The sampling rate of the signal (in Hz, i.e., samples per second).
    check_height_outlier : bool, optional
        Whether to check for height outliers in the local signal.
    k_sample_ratio : float, optional
        Ratio of samples to consider when checking for height outliers.
    use_prominence : bool, optional
        Whether to use peak prominence when checking for height outliers.
    **kwargs
        Additional keyword arguments passed to the `get_local_hb_sig` function.

    Returns
    -------
    np.ndarray
        Array of timestamps corresponding to the corrected peak times.
    """
    if sig_time is None:
        sig_time = sampling_rate_to_sig_time(sig=sig, sampling_rate=sampling_rate)
    else:
        sampling_rate = sig_time_to_sampling_rate(sig_time=sig_time)

    new_peak_time = []

    if check_height_outlier:
        peak_height = sig[timestamp_to_samp(peak_time, sampling_rate, sig_time)]

    for peak in peak_time:
        hb_sig, hb_sig_time = get_local_hb_sig(
            peak,
            sig=sig,
            sig_time=sig_time,
            sampling_rate=sampling_rate,
            **get_func_kwargs(get_local_hb_sig, **kwargs)
        )

        if check_height_outlier:
            if k_sample_ratio == 0:
                k = 1
            else:
                k = int(k_sample_ratio * len(hb_sig))

            if use_prominence:
                local_peaks, _ = scipy.signal.find_peaks(hb_sig)
                local_prominence = scipy.signal.peak_prominences(hb_sig, local_peaks)[0]
                potential_peaks_index = local_peaks[argtop_k(local_prominence, k=k)]
            else:
                potential_peaks_index = argtop_k(hb_sig, k=k)

            peak_is_outlier = True
            i = 0
            current_peak_index = np.nan

            while peak_is_outlier and i < len(potential_peaks_index):
                current_peak_index = potential_peaks_index[i]
                current_peak_height = hb_sig[current_peak_index]
                peak_height_with_current = peak_height.copy()
                peak_height_with_current = np.insert(
                    peak_height_with_current, 0, current_peak_height
                )

                # Having a fit and predict class like sklearn estimator
                # would probably make this faster
                peak_is_outlier = nk.find_outliers(peak_height_with_current)[0]
                i += 1

                # Alternatively instead of iterating through can make
                # sure that there are no two candidate peaks that are
                # spaced apart far enough to be S1 and S2
            if np.isnan(current_peak_index) or peak_is_outlier:
                new_peak = peak
            else:
                new_peak = hb_sig_time[current_peak_index]
        else:
            if len(hb_sig) > 1:
                if use_prominence:
                    local_peaks, _ = scipy.signal.find_peaks(hb_sig)
                    prominences = scipy.signal.peak_prominences(hb_sig, local_peaks)[0]
                    new_peak = hb_sig_time[local_peaks[np.argmax(prominences)]]
                else:
                    new_peak = hb_sig_time[np.argmax(hb_sig)]
            else:
                new_peak = peak

        new_peak_time.append(new_peak)

    new_peak_time = np.array(new_peak_time)
    return new_peak_time
