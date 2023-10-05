from typing import Optional, Tuple

import neurokit2 as nk
import numpy as np
import scipy
from neurokit2.hrv.intervals_utils import _intervals_successive
from scipy import interpolate

from ..misc.misc_utils import argtop_k, get_func_kwargs
from .preprocessing_utils import (
    sampling_rate_to_sig_time,
    sig_time_to_sampling_rate,
    timestamp_to_samp,
)


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


def fixpeaks_by_height(
    peak_time: np.ndarray,
    sig_info: dict = None,
    clean_sig_info: dict = None,
    sig_name: str = "zephyr_ecg",
    time_boundaries: dict = None,
) -> np.ndarray:
    """
    Fix detected peaks based on their heights.

    Parameters
    ----------
    peak_time : np.ndarray
        Array of peak times to be fixed.
    sig_info : dict, optional
        Information about the signal containing the peaks.
    clean_sig_info : dict, optional
        Information about the cleaned signal.
    sig_name : str, optional
        Name of the signal.
    time_boundaries : dict, optional
        Time boundaries for peak detection.

    Returns
    -------
    np.ndarray
        Array of fixed peak times.
    """
    new_peak_time = []
    if time_boundaries is None:
        time_boundaries = {
            "before_peak_clean": 0.1,
            "after_peak_clean": 0.1,
            "before_peak_raw": (0.005,),
            "after_peak_raw": 0.005 if sig_name == "zephyr_ecg" else 0.001,
        }

    for seg_peak_time in peak_time:
        seg_sig = sig_info["sig"]
        seg_sig_time = sig_info["time"]
        sampling_rate = sig_info["sampling_rate"]

        if clean_sig_info is None:
            seg_clean_sig = nk.signal_filter(
                seg_sig,
                sampling_rate=sampling_rate,
                lowcut=0.5,
                highcut=8,
                method="butterworth",
                order=2,
            )
            seg_clean_sig_time = seg_sig_time
        else:
            seg_clean_sig = clean_sig_info["sig"]
            seg_clean_sig_time = clean_sig_info["time"]

        new_seg_clean_peak_time = find_local_hb_peaks(
            peak_time=[seg_peak_time],
            sig=seg_clean_sig,
            sig_time=seg_clean_sig_time,
            time_before_peak=time_boundaries["before_peak_clean"],
            time_after_peak=time_boundaries["after_peak_clean"],
        )

        new_seg_peak_time = find_local_hb_peaks(
            peak_time=new_seg_clean_peak_time,
            sig=seg_sig,
            sig_time=seg_sig_time,
            time_before_peak=time_boundaries["before_peak_raw"],
            time_after_peak=time_boundaries["after_peak_raw"],
        )

        new_peak_time.append(new_seg_peak_time)

    return np.concatenate(new_peak_time)


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


def clean_hb_signal(
    sig: np.ndarray, sampling_rate: int, clean_method: str, highcut: int
) -> np.ndarray:
    """
    Clean the input signal using specified method.

    Parameters
    ----------
    sig : np.ndarray
        The input signal.
    sampling_rate : int
        The sampling rate of the signal.
    clean_method : str
        The method for cleaning the signal.
    highcut : int
        Highcut frequency for signal filtering.

    Returns
    -------
    np.ndarray
        The cleaned signal.
    """

    if clean_method == "own_filt":
        clean_sig = nk.signal_filter(
            sig,
            sampling_rate=sampling_rate,
            lowcut=0.5,
            highcut=highcut,
            method="butterworth",
            order=2,
        )
    else:
        clean_sig = nk.ecg_clean(sig, method="engzeemod2012")

    return clean_sig


def resample_hb_signal(
    clean_sig: np.ndarray,
    sig_time: np.ndarray,
    sampling_rate: int,
    new_sampling_rate: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Resample the cleaned signal to a new sampling rate.

    Parameters
    ----------
    clean_sig : np.ndarray
        The cleaned signal.
    sig_time : np.ndarray
        The time values corresponding to the signal.
    sampling_rate : int
        The original sampling rate of the signal.
    new_sampling_rate : int
        The target sampling rate.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        The resampled signal and corresponding time values.
    """
    div = sampling_rate / new_sampling_rate
    resampled_clean_sig, resampled_clean_sig_time = scipy.signal.resample(
        clean_sig, num=int(len(clean_sig) / div), t=sig_time
    )

    return resampled_clean_sig, resampled_clean_sig_time


def clean_and_resample_signal(
    sig: np.ndarray,
    sig_time: np.ndarray,
    sampling_rate: int,
    clean_method: str,
    highcut: int,
    new_sampling_rate: int = 100,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Clean and resample the input signal.

    Parameters
    ----------
    sig : np.ndarray
        The input signal.
    sig_time : np.ndarray
        The time values corresponding to the signal.
    sampling_rate : int
        The sampling rate of the signal.
    clean_method : str
        The method for cleaning the signal.
    highcut : int
        Highcut frequency for signal filtering.
    new_sampling_rate : int, optional
        The target sampling rate.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Tuple containing the cleaned signal and corresponding time values after resampling.
    """
    # Clean the signal
    clean_sig = clean_hb_signal(sig, sampling_rate, clean_method, highcut)

    # Resample the cleaned signal
    resampled_clean_sig, resampled_clean_sig_time = resample_hb_signal(
        clean_sig, sig_time, sampling_rate, new_sampling_rate=new_sampling_rate
    )

    return resampled_clean_sig, resampled_clean_sig_time
