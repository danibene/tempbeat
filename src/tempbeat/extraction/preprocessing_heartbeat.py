from typing import Optional, Tuple

import neurokit2 as nk
import numpy as np
import scipy
from neurokit2.hrv.intervals_utils import _intervals_successive

from .segmentation import find_local_hb_peaks


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
