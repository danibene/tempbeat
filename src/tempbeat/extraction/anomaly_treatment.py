from typing import Tuple

import neurokit2 as nk
import numpy as np

from ..utils.misc_utils import a_moving_average, argtop_k
from ..utils.timestamp import samp_to_timestamp, timestamp_to_samp
from .interval_conversion import peak_time_to_rri, rri_to_peak_time
from .mod_fixpeaks import signal_fixpeaks
from .segmentation import find_local_hb_peaks


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
    if time_boundaries is None:
        time_boundaries = {
            "before_peak_clean": 0.1,
            "after_peak_clean": 0.1,
            "before_peak_raw": (0.005,),
            "after_peak_raw": 0.005 if sig_name == "zephyr_ecg" else 0.001,
        }

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

    new_peak_time = np.empty(len(peak_time), dtype=object)

    for i, seg_peak_time in enumerate(peak_time):
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

        new_peak_time[i] = new_seg_peak_time

    return np.concatenate(new_peak_time.tolist())


def remove_anomalies_from_corr_peaks(
    peak_time_from_corr: np.ndarray,
    sig_time: np.ndarray,
    corrs: np.ndarray,
    corr_ind: int,
    min_n_confident_peaks: int,
    corr_heights: np.ndarray,
    thr_corr_height: float,
    relative_rri_for_temp_min: float,
    relative_rri_for_temp_max: float,
    min_bpm: int = 40,
    max_bpm: int = 200,
    max_time_after_last_peak: int = 5,
    use_rri_to_peak_time: bool = True,
    move_average_rri_window: int = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Handle anomalies in the peak times extracted from the correlation signal.

    Parameters
    ----------
    peak_time_from_corr : np.ndarray
        Peak times extracted from the correlation signal.
    sig_time : np.ndarray
        The time values corresponding to the original signal.
    corrs : np.ndarray
        The correlation signal.
    corr_ind : int
        The indices of the correlation signal corresponding to peaks.
    min_n_confident_peaks : int
        Minimum number of confident peaks.
    corr_heights : np.ndarray
        Heights of the peaks extracted from the correlation signal.
    thr_corr_height : float
        Threshold for correlation height.
    relative_rri_for_temp_min : float
        Minimum relative RRI.
    relative_rri_for_temp_max : float
        Maximum relative RRI.
    min_bpm : int, optional
        The minimum heart rate in beats per minute. Defaults to 40.
    max_bpm : int, optional
        The maximum heart rate in beats per minute. Defaults to 200.
    max_time_after_last_peak : int, optional
        Maximum time after the last peak.
    use_rri_to_peak_time : bool, optional
        Whether to use RRI for peak time.
    move_average_rri_window : int, optional
        Window size for moving average of RRI.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Tuple containing the peak times after height-based filtering and R-R interval based filtering.
    """
    peak_time_from_corr_height_filtered = peak_time_from_corr[
        (nk.standardize(corr_heights, robust=True) >= thr_corr_height)
    ]
    if len(peak_time_from_corr_height_filtered) < min_n_confident_peaks:
        peak_time_from_corr_height_filtered = peak_time_from_corr[
            argtop_k(corr_heights, k=min_n_confident_peaks)
        ]

    rri, rri_time = peak_time_to_rri(
        peak_time_from_corr_height_filtered,
        min_rri=60000 / max_bpm,
        max_rri=60000 / min_bpm,
    )

    if move_average_rri_window is not None:
        stand = np.abs(nk.standardize(rri, robust=True))
        stand = a_moving_average(stand, N=move_average_rri_window)
        relative_rri_for_temp = np.mean(
            np.abs([relative_rri_for_temp_min, relative_rri_for_temp_max])
        )
        anomalies = np.invert(stand <= relative_rri_for_temp)
    else:
        anomalies = np.invert(
            (nk.standardize(rri, robust=True) >= relative_rri_for_temp_min)
            & (nk.standardize(rri, robust=True) <= relative_rri_for_temp_max)
        )

    if len(rri_time) - len(rri_time[anomalies]) < min_n_confident_peaks - 1:
        anomalies = argtop_k(
            np.abs(nk.standardize(rri, robust=True)),
            k=len(rri) - min_n_confident_peaks,
        )

    rri_time[anomalies] = np.nan
    rri[anomalies] = np.nan
    if use_rri_to_peak_time:
        peak_time_from_corr_rri_filtered = rri_to_peak_time(rri=rri, rri_time=rri_time)
    else:
        peak_time_from_corr_rri_filtered = np.concatenate(
            (np.array([peak_time_from_corr_height_filtered[0]]), rri_time)
        )

    min_last_peak_time = np.max(sig_time) - max_time_after_last_peak
    if np.max(peak_time_from_corr_rri_filtered) < min_last_peak_time:
        new_last_peak = peak_time_from_corr[
            argtop_k(corrs[corr_ind][peak_time_from_corr > min_last_peak_time], k=1)
        ]
        peak_time_from_corr_rri_filtered = np.append(
            peak_time_from_corr_rri_filtered, new_last_peak
        )
    return peak_time_from_corr_height_filtered, peak_time_from_corr_rri_filtered


def _get_added_kept_peak_time(
    fixed_peak_time: np.ndarray = None,
    original_kept_peak_time: np.ndarray = None,
    dec: int = 2,
) -> Tuple[np.ndarray, np.ndarray]:
    added_peak_time = np.array(
        [
            peak
            for peak in fixed_peak_time
            if np.round(peak, dec) not in np.round(original_kept_peak_time, dec)
        ]
    )
    kept_peak_time = np.array(
        [
            peak
            for peak in fixed_peak_time
            if np.round(peak, dec) in np.round(original_kept_peak_time, dec)
        ]
    )

    return added_peak_time, kept_peak_time


def fix_final_peaks(
    peak_time_from_corr_rri_filtered: np.ndarray,
    orig_sig: np.ndarray,
    orig_sig_time: np.ndarray,
    orig_sampling_rate: int,
    corrs: np.ndarray,
    corr_times: np.ndarray,
    sampling_rate: int,
    fix_interpl_peaks_by_height: bool,
    fix_added_interpl_peaks_by_height: bool,
    fixpeaks_by_height_time_boundaries: float,
    k_nearest_intervals: int,
    n_nan_estimation_method: str,
    interpolate_args: dict,
    min_bpm: int = 40,
    max_bpm: int = 200,
) -> np.ndarray:
    """
    Fix filtered peaks.

    Parameters
    ----------
    peak_time_from_corr_rri_filtered : np.ndarray
        Peak times after initial filtering.
    orig_sig : np.ndarray
        The original signal.
    orig_sig_time : np.ndarray
        The time values corresponding to the original signal.
    orig_sampling_rate : int
        The original sampling rate of the signal.
    corrs : np.ndarray
        The correlation signal.
    corr_times : np.ndarray
        The time values corresponding to the correlation signal.
    sampling_rate : int
        The sampling rate of the signal.
    fix_interpl_peaks_by_height : bool
        Whether to fix interpolated peaks by height.
    fix_added_interpl_peaks_by_height : bool
        Whether to fix added interpolated peaks by height.
    fixpeaks_by_height_time_boundaries : float
        Time boundaries for fixing peaks by height.
    k_nearest_intervals : int
        Number of nearest intervals for interpolation.
    n_nan_estimation_method : str
        Method for estimating number of NaNs.
    interpolate_args : dict
        Arguments for interpolation.
    min_bpm : int, optional
        The minimum heart rate in beats per minute. Defaults to 40.
    max_bpm : int, optional
        The maximum heart rate in beats per minute. Defaults to 200.
    """
    rri, _ = peak_time_to_rri(
        peak_time_from_corr_rri_filtered,
        min_rri=60000 / max_bpm,
        max_rri=60000 / min_bpm,
    )

    new_peaks = timestamp_to_samp(
        peak_time_from_corr_rri_filtered, sig_time=orig_sig_time
    )

    fixed_new_peaks = signal_fixpeaks(
        new_peaks,
        method="neurokit",
        sampling_rate=sampling_rate,
        interval_min=np.nanmin(rri) / 1000,
        interval_max=np.nanmax(rri) / 1000,
        robust=True,
        k_nearest_intervals=k_nearest_intervals,
        n_nan_estimation_method=n_nan_estimation_method,
        interpolate_args=interpolate_args,
    )
    fixed_peak_time = samp_to_timestamp(fixed_new_peaks, sig_time=orig_sig_time)
    if fix_interpl_peaks_by_height:
        final_peak_time = fixpeaks_by_height(
            samp_to_timestamp(fixed_new_peaks, sig_time=orig_sig_time),
            sig_info={
                "time": corr_times,
                "sig": corrs,
                "sampling_rate": sampling_rate,
            },
            time_boundaries=fixpeaks_by_height_time_boundaries,
        )
    elif fix_added_interpl_peaks_by_height:
        added_peak_time, _ = _get_added_kept_peak_time(
            fixed_peak_time=fixed_peak_time,
            original_kept_peak_time=peak_time_from_corr_rri_filtered,
        )
        height_fixed_added_peak_time = fixpeaks_by_height(
            added_peak_time,
            sig_info={
                "time": orig_sig_time,
                "sig": orig_sig,
                "sampling_rate": orig_sampling_rate,
            },
            clean_sig_info={
                "time": corr_times,
                "sig": corrs,
                "sampling_rate": sampling_rate,
            },
            time_boundaries=fixpeaks_by_height_time_boundaries,
        )
        final_peak_time = np.sort(
            np.concatenate(
                (peak_time_from_corr_rri_filtered, height_fixed_added_peak_time)
            )
        )
    else:
        final_peak_time = fixed_peak_time

    return final_peak_time
