import pathlib
from typing import Tuple

import neurokit2 as nk
import numpy as np
import scipy

from ..misc.misc_utils import argtop_k, write_dict_to_json
from .mod_fixpeaks import signal_fixpeaks
from .preprocessing_heartbeat import (
    find_anomalies,
    fixpeaks_by_height,
    get_local_hb_sig,
    peak_time_to_rri,
    rri_to_peak_time,
)
from .preprocessing_utils import (
    a_moving_average,
    norm_corr,
    roll_func,
    samp_to_timestamp,
    timestamp_to_samp,
)


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


def extract_potential_peaks_for_template_estimation(
    resampled_clean_sig: np.ndarray,
    resampled_clean_sig_time: np.ndarray,
    sampling_rate: int,
    relative_peak_height_for_temp_min: float,
    relative_peak_height_for_temp_max: float,
    min_n_peaks_for_temp_confident: int,
    min_bpm: int = 40,
) -> Tuple[np.ndarray, float, float]:
    """
    Extract potential peaks from the cleaned signal.

    Parameters
    ----------
    resampled_clean_sig : np.ndarray
        The cleaned signal.
    resampled_clean_sig_time : np.ndarray
        The time values corresponding to the cleaned signal.
    sampling_rate : int
        The sampling rate of the signal.
    relative_peak_height_for_temp_min : float
        Minimum relative peak height for peaks used for computing template.
    relative_peak_height_for_temp_max : float
        Maximum relative peak height for temporal confidence.
    min_n_peaks_for_temp_confident : int
        Minimum number of peaks to consider for computing template.
    min_bpm : int, optional
        The minimum heart rate in beats per minute.

    Returns
    -------
    Tuple[np.ndarray, float, float]
        Tuple containing potential peaks, minimum height, and maximum height.
    """
    window_time = 60 / min_bpm
    window = int(sampling_rate * window_time) * 2
    potential_peaks = roll_func(resampled_clean_sig, window=window, func=np.max)
    stand_peaks = nk.standardize(potential_peaks, robust=True)
    peaks_no_outliers = potential_peaks[
        (stand_peaks <= relative_peak_height_for_temp_max)
        & (stand_peaks >= relative_peak_height_for_temp_min)
    ]
    if len(peaks_no_outliers) < min_n_peaks_for_temp_confident:
        peaks_no_outliers = potential_peaks[
            argtop_k(-1 * np.abs(stand_peaks), k=min_n_peaks_for_temp_confident)
        ]

    height_min = np.min(peaks_no_outliers)
    height_max = np.max(peaks_no_outliers)
    peak_info = nk.signal_findpeaks(resampled_clean_sig)
    good_peaks = peak_info["Peaks"][
        (resampled_clean_sig[peak_info["Peaks"]] >= height_min)
        & (resampled_clean_sig[peak_info["Peaks"]] <= height_max)
    ]
    peak_time_for_temp = samp_to_timestamp(
        good_peaks, sig_time=resampled_clean_sig_time
    )
    return peak_time_for_temp, height_min, height_max


def compute_rri_and_handle_anomalies_for_template(
    potential_peak_time_for_temp: np.ndarray,
    relative_rri_for_temp_min: float,
    relative_rri_for_temp_max: float,
    move_average_rri_window: int,
    min_bpm: int = 40,
    max_bpm: int = 200,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute R-R intervals (RRI) and handle anomalies.

    Parameters
    ----------
    potential_peak_time_for_temp : np.ndarray
        Peak times used for computing RRI.
    relative_rri_for_temp_min : float
        Minimum relative RRI for temporal confidence.
    relative_rri_for_temp_max : float
        Maximum relative RRI for temporal confidence.
    move_average_rri_window : int
        Window size for moving average of RRI.
    min_bpm : int, optional
        The minimum heart rate in beats per minute. Defaults to 40.
    max_bpm : int, optional
        The maximum heart rate in beats per minute. Defaults to 200.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Tuple containing computed RRI and corresponding time values.
    """
    rri, rri_time = peak_time_to_rri(
        potential_peak_time_for_temp,
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

    rri_time[anomalies] = np.nan
    rri[anomalies] = np.nan

    return rri, rri_time


def generate_template(
    peak_time_for_temp_confident: np.ndarray,
    resampled_clean_sig: np.ndarray,
    resampled_clean_sig_time: np.ndarray,
    new_sampling_rate: int,
    temp_time_before_peak: float,
    temp_time_after_peak: float,
) -> np.ndarray:
    """
    Generate template based on peak times and cleaned signal.

    Parameters
    ----------
    peak_time_for_temp_confident : np.ndarray
        Peak times used for template generation.
    resampled_clean_sig : np.ndarray
        Cleaned signal after resampling.
    resampled_clean_sig_time : np.ndarray
        Time values corresponding to the cleaned signal after resampling.
    new_sampling_rate : int
        The new sampling rate after resampling.
    temp_time_before_peak : float
        Time before peak for template extraction.
    temp_time_after_peak : float
        Time after peak for template extraction.

    Returns
    -------
    np.ndarray
        The generated template.
    """
    templates = np.array([])

    for peak in peak_time_for_temp_confident:
        # Extract local heartbeat signal around the peak
        hb_sig, _ = get_local_hb_sig(
            peak=peak,
            sig=resampled_clean_sig,
            sig_time=resampled_clean_sig_time,
            sampling_rate=new_sampling_rate,
            time_before_peak=temp_time_before_peak,
            time_after_peak=temp_time_after_peak,
        )
        # TODO: make the window length based on samples rather than times
        # so that there aren't rounding errors

        # Define the template length in terms of samples
        template_len = int(
            temp_time_before_peak * new_sampling_rate
            + temp_time_after_peak * new_sampling_rate
            - 2
        )

        if len(hb_sig) > template_len:
            if len(templates) < 1:
                templates = hb_sig[:template_len]
            else:
                templates = np.vstack((templates, hb_sig[:template_len]))

    med_template = np.nanmedian(np.array(templates), axis=0)

    return med_template


def correlate_templates_with_signal(
    med_template: np.ndarray,
    resampled_clean_sig: np.ndarray,
    resampled_clean_sig_time: np.ndarray,
    new_sampling_rate: int,
    temp_time_before_peak: float,
    temp_time_after_peak: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Correlate templates with the cleaned signal.

    Parameters
    ----------
    med_template : np.ndarray
        Median template computed from generated templates.
    resampled_clean_sig : np.ndarray
        Cleaned signal after resampling.
    resampled_clean_sig_time : np.ndarray
        Time values corresponding to the cleaned signal after resampling.
    new_sampling_rate : int
        The new sampling rate after resampling.
    temp_time_before_peak : float
        Time before peak for template extraction.
    temp_time_after_peak : float
        Time after peak for template extraction.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Tuple containing correlation values and corresponding time values.
    """
    corrs = []
    corr_times = []
    sig = resampled_clean_sig
    sig_time = resampled_clean_sig_time
    sampling_rate = new_sampling_rate
    for time in sig_time:
        hb_sig, _ = get_local_hb_sig(
            time,
            sig,
            sig_time=sig_time,
            sampling_rate=sampling_rate,
            time_before_peak=temp_time_before_peak,
            time_after_peak=temp_time_after_peak,
        )
        if len(hb_sig) >= len(med_template):
            if len(hb_sig) > len(med_template):
                hb_sig = hb_sig[: len(med_template)]
            corr = norm_corr(hb_sig, med_template)[0]
            corrs.append(corr)
            corr_times.append(time)
    corrs = np.array(corrs)
    corr_times = np.array(corr_times)

    return corrs, corr_times


def extract_potential_peaks_from_correlation(
    corrs: np.ndarray,
    corr_times: np.ndarray,
    sampling_rate: int,
    corr_peak_extraction_method: str,
    fix_corr_peaks_by_height: bool,
    fixpeaks_by_height_time_boundaries: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract potential peaks from the correlation signal.

    Parameters
    ----------
    corrs : np.ndarray
        The correlation signal.
    corr_times : np.ndarray
        The time values corresponding to the correlation signal.
    sampling_rate : int
        The sampling rate of the signal.
    corr_peak_extraction_method : str
        Peak extraction method used for correlation signal.
    fix_corr_peaks_by_height : bool
        Whether to fix correlated peaks by height.
    fixpeaks_by_height_time_boundaries : float
        Time boundaries for fixing peaks by height.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        Tuple containing potential peaks, corresponding heights, and indices.
    """
    # can use the neurokit fixpeaks afterwards with the large missing values
    # max time_before_peak/time_after_peak in fix_local hb sig with clean can be based
    # on expected deviation from the linear interpolation i.e. mean
    # then do final fix_local on raw data to account for resampling to 100 Hz
    if corr_peak_extraction_method == "nk_ecg_peaks_nabian2018":
        _, processed_info = nk.ecg_peaks(
            corrs, method="nabian2018", sampling_rate=sampling_rate
        )
    elif corr_peak_extraction_method == "nk_ecg_process_kalidas2017":
        _, processed_info = nk.ecg_process(
            corrs, sampling_rate=sampling_rate, method="kalidas2017"
        )
    else:
        _, processed_info = nk.ecg_process(corrs, sampling_rate=sampling_rate)
    peak_key = [key for key in list(processed_info.keys()) if "Peak" in key][0]
    ind = [processed_info[peak_key]][0].astype(int)
    peak_time_from_corr = corr_times[ind]
    if fix_corr_peaks_by_height:
        peak_time_from_corr_old = peak_time_from_corr.copy()
        peak_time_from_corr = fixpeaks_by_height(
            peak_time_from_corr_old,
            sig_info={
                "time": corr_times,
                "sig": corrs,
                "sampling_rate": sampling_rate,
            },
            time_boundaries=fixpeaks_by_height_time_boundaries,
        )
        corr_heights = corrs[
            timestamp_to_samp(
                peak_time_from_corr, sampling_rate=sampling_rate, sig_time=corr_times
            )
        ]
    else:
        corr_heights = corrs[ind]
    return peak_time_from_corr, corr_heights, ind


def handle_anomalies_in_peak_time_from_corr(
    peak_time_from_corr: np.ndarray,
    sig_time: np.ndarray,
    resampled_clean_sig: np.ndarray,
    resampled_clean_sig_time: np.ndarray,
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
    use_rri_to_peak_time: bool = False,
    find_anomalies_threshold: float = None,
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
    resampled_clean_sig : np.ndarray
        Cleaned signal after resampling.
    resampled_clean_sig_time : np.ndarray
        Time values corresponding to the cleaned signal after resampling.
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
    find_anomalies_threshold : float, optional
        Threshold for finding anomalies.
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

    # here is where the peak fixing for the higher fs signal could go?
    if find_anomalies_threshold is None:
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
            peak_time_from_corr_rri_filtered = rri_to_peak_time(
                rri=rri, rri_time=rri_time
            )
        else:
            peak_time_from_corr_rri_filtered = np.concatenate(
                (np.array([peak_time_from_corr_height_filtered[0]]), rri_time)
            )
    else:
        anomalies_score = find_anomalies(
            peak_time_from_corr_height_filtered,
            sig_info={"sig": resampled_clean_sig, "time": resampled_clean_sig_time},
        )
        anomalies = anomalies_score > find_anomalies_threshold
        if (
            len(peak_time_from_corr_height_filtered)
            - len(peak_time_from_corr_height_filtered[anomalies])
            < min_n_confident_peaks
        ):
            anomalies = argtop_k(
                anomalies_score,
                k=len(peak_time_from_corr_height_filtered) - min_n_confident_peaks,
            )
        peak_time_from_corr_rri_filtered = peak_time_from_corr_height_filtered.copy()
        peak_time_from_corr_rri_filtered[anomalies] = np.nan
        peak_time_from_corr_rri_filtered = peak_time_from_corr_rri_filtered[
            np.isfinite(peak_time_from_corr_rri_filtered)
        ]

    min_last_peak_time = np.max(sig_time) - max_time_after_last_peak
    if np.max(peak_time_from_corr_rri_filtered) < min_last_peak_time:
        new_last_peak = peak_time_from_corr[
            argtop_k(corrs[corr_ind][peak_time_from_corr > min_last_peak_time], k=1)
        ]
        peak_time_from_corr_rri_filtered = np.append(
            peak_time_from_corr_rri_filtered, new_last_peak
        )
    return peak_time_from_corr_height_filtered, peak_time_from_corr_rri_filtered


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
        dec = 1
        fixed_peak_time = samp_to_timestamp(fixed_new_peaks, sig_time=orig_sig_time)
        added_peak_time = np.array(
            [
                peak
                for peak in fixed_peak_time
                if np.round(peak, 1)
                not in np.round(peak_time_from_corr_rri_filtered, dec)
            ]
        )
        kept_peak_time = np.array(
            [
                peak
                for peak in fixed_peak_time
                if np.round(peak, 1) in np.round(peak_time_from_corr_rri_filtered, dec)
            ]
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
            np.concatenate((kept_peak_time, height_fixed_added_peak_time))
        )
    else:
        final_peak_time = samp_to_timestamp(fixed_new_peaks, sig_time=orig_sig_time)

    return final_peak_time


def export_debug_info(
    resampled_clean_sig: np.ndarray,
    resampled_clean_sig_time: np.ndarray,
    new_sampling_rate: int,
    height_min: float,
    height_max: float,
    potential_peak_time_for_temp: np.ndarray,
    peak_time_for_temp_confident: np.ndarray,
    med_template: np.ndarray,
    corrs: np.ndarray,
    corr_times: np.ndarray,
    peak_time_from_corr: np.ndarray,
    peak_time_from_corr_height_filtered: np.ndarray,
    peak_time_from_corr_rri_filtered: np.ndarray,
    final_peak_time: np.ndarray,
    debug_out_path: str = None,
):
    """
    Export debug information.

    Parameters
    ----------
    resampled_clean_sig : np.ndarray
        Cleaned signal after resampling.
    resampled_clean_sig_time : np.ndarray
        Time values corresponding to the cleaned signal after resampling.
    new_sampling_rate : int
        The new sampling rate after resampling.
    height_min : float
        Minimum height of peaks.
    height_max : float
        Maximum height of peaks.
    potential_peak_time_for_temp : np.ndarray
        Peak times used for template generation.
    peak_time_for_temp_confident : np.ndarray
        Peak times used for template generation after height-based filtering.
    med_template : np.ndarray
        Median template computed from generated templates.
    corrs : np.ndarray
        The correlation signal.
    corr_times : np.ndarray
        The time values corresponding to the correlation signal.
    peak_time_from_corr : np.ndarray
        Peak times extracted from the correlation signal.
    peak_time_from_corr_height_filtered : np.ndarray
        Peak times extracted from the correlation signal after height-based filtering.
    peak_time_from_corr_rri_filtered : np.ndarray
        Peak times after initial filtering.
    final_peak_time : np.ndarray
        Final peak times.
    debug_out_path : str, optional
        Path to save debug information.
    """
    debug_out = {}
    debug_out["resampled_clean_sig"] = resampled_clean_sig
    debug_out["resampled_clean_sig_time"] = resampled_clean_sig_time
    debug_out["new_sampling_rate"] = new_sampling_rate
    debug_out["height_min"] = height_min
    debug_out["height_max"] = height_max
    debug_out["peak_time_for_temp"] = potential_peak_time_for_temp
    debug_out["peak_time_for_temp_confident"] = peak_time_for_temp_confident
    debug_out["med_template"] = med_template
    debug_out["corrs"] = corrs
    debug_out["corr_times"] = corr_times
    debug_out["peak_time_from_corr"] = peak_time_from_corr
    debug_out[
        "peak_time_from_corr_height_filtered"
    ] = peak_time_from_corr_height_filtered
    debug_out["peak_time_from_corr_rri_filtered"] = peak_time_from_corr_rri_filtered
    debug_out["final_peak_time"] = final_peak_time
    if debug_out_path is None:
        return final_peak_time, debug_out
    else:
        time_str = "_" + str(int(np.min(resampled_clean_sig_time)))
        this_debug_out_path = str(
            pathlib.Path(
                pathlib.Path(debug_out_path).parent,
                pathlib.Path(debug_out_path).stem,
                pathlib.Path(debug_out_path).stem + time_str + ".json",
            )
        )
        write_dict_to_json(debug_out, json_path=this_debug_out_path)
        return final_peak_time


def temp_hb_extract(
    sig: np.ndarray,
    sig_time: np.ndarray = None,
    sampling_rate: int = 1000,
    min_bpm: int = 40,
    max_bpm: int = 200,
    thr_corr_height: float = -2.5,
    min_n_peaks_for_temp_confident: int = 5,
    relative_peak_height_for_temp_min: float = -2,
    relative_peak_height_for_temp_max: float = 2,
    relative_rri_for_temp_min: float = -2,
    relative_rri_for_temp_max: float = 2,
    min_n_confident_peaks: int = 5,
    max_time_after_last_peak: int = 5,
    clean_method: str = "own_filt",
    highcut: int = 8,
    fix_corr_peaks_by_height: bool = False,
    fix_interpl_peaks_by_height: bool = False,
    fix_added_interpl_peaks_by_height: bool = False,
    corr_peak_extraction_method: str = "nk_ecg_process",
    k_nearest_intervals: int = None,
    n_nan_estimation_method: str = "floor",
    interpolate_args: dict = {"method": "linear"},
    temp_time_before_peak: float = 0.3,
    temp_time_after_peak: float = 0.3,
    fixpeaks_by_height_time_boundaries: float = None,
    use_rri_to_peak_time: bool = False,
    find_anomalies_threshold: float = None,
    move_average_rri_window: int = None,
    output_format: str = "only_final",
    debug_out_path: str = None,
) -> np.ndarray:
    """
    Template-based heartbeat extraction.

    Parameters
    ----------
    sig : np.ndarray
        The input signal.
    sig_time : np.ndarray, optional
        The time values corresponding to the signal.
    sampling_rate : int, optional
        The sampling rate of the signal.
    min_bpm : int, optional
        The minimum heart rate in beats per minute.
    max_bpm : int, optional
        The maximum heart rate in beats per minute.
    thr_corr_height : float, optional
        Threshold for correlation height.
    min_n_peaks_for_temp_confident : int, optional
        Minimum number of peaks to consider for computing template.
    relative_peak_height_for_temp_min : float, optional
        Minimum relative peak height for peaks used for computing template.
    relative_peak_height_for_temp_max : float, optional
        Maximum relative peak height for temporal confidence.
    relative_rri_for_temp_min : float, optional
        Minimum relative RRI for temporal confidence.
    relative_rri_for_temp_max : float, optional
        Maximum relative RRI for temporal confidence.
    min_n_confident_peaks : int, optional
        Minimum number of confident peaks.
    max_time_after_last_peak : int, optional
        Maximum time after the last peak.
    clean_method : str, optional
        The method for cleaning the signal.
    highcut : int, optional
        Highcut frequency for signal filtering.
    fix_corr_peaks_by_height : bool, optional
        Whether to fix correlated peaks by height.
    fix_interpl_peaks_by_height : bool, optional
        Whether to fix interpolated peaks by height.
    fix_added_interpl_peaks_by_height : bool, optional
        Whether to fix added interpolated peaks by height.
    corr_peak_extraction_method : str, optional
        Peak extraction method used for correlation signal.
    k_nearest_intervals : int, optional
        The number of nearest intervals.
    n_nan_estimation_method : str, optional
        Method for estimating NaNs.
    interpolate_args : dict, optional
        Arguments for interpolation.
    temp_time_before_peak : float, optional
        Time before peak for template extraction.
    temp_time_after_peak : float, optional
        Time after peak for template extraction.
    fixpeaks_by_height_time_boundaries : float, optional
        Time boundaries for fixing peaks by height.
    use_rri_to_peak_time : bool, optional
        Whether to use RRI for peak time.
    find_anomalies_threshold : float, optional
        Threshold for finding anomalies.
    move_average_rri_window : int, optional
        Window size for moving average of RRI.
    output_format : str, optional
        Output format ('only_final' or 'full').
    debug_out_path : str, optional
        Path for debug output.

    Returns
    -------
    np.ndarray
        The extracted heartbeat times.
    """
    orig_sig = sig.copy()
    orig_sig_time = sig_time.copy()
    orig_sampling_rate = sampling_rate

    # Clean and resample the signal
    new_sampling_rate = 100
    resampled_clean_sig, resampled_clean_sig_time = clean_and_resample_signal(
        sig=sig,
        sig_time=sig_time,
        sampling_rate=sampling_rate,
        clean_method=clean_method,
        highcut=highcut,
        new_sampling_rate=new_sampling_rate,
    )

    # Extract potential peaks for template
    (
        potential_peak_time_for_temp,
        height_min,
        height_max,
    ) = extract_potential_peaks_for_template_estimation(
        resampled_clean_sig,
        resampled_clean_sig_time,
        new_sampling_rate,
        relative_peak_height_for_temp_min,
        relative_peak_height_for_temp_max,
        min_n_peaks_for_temp_confident,
    )

    # Compute RRI and handle anomalies
    rri, rri_time = compute_rri_and_handle_anomalies_for_template(
        potential_peak_time_for_temp,
        relative_rri_for_temp_min,
        relative_rri_for_temp_max,
        move_average_rri_window,
        min_bpm=min_bpm,
        max_bpm=max_bpm,
    )

    if use_rri_to_peak_time:
        peak_time_for_temp_confident = rri_to_peak_time(rri=rri, rri_time=rri_time)
    else:
        peak_time_for_temp_confident = np.concatenate(
            (
                np.array([potential_peak_time_for_temp[0]]),
                rri_time[(np.abs(nk.standardize(rri, robust=True)) < 2)],
            )
        )

    med_template = generate_template(
        peak_time_for_temp_confident,
        resampled_clean_sig,
        resampled_clean_sig_time,
        new_sampling_rate,
        temp_time_before_peak,
        temp_time_after_peak,
    )

    corrs, corr_times = correlate_templates_with_signal(
        med_template,
        resampled_clean_sig,
        resampled_clean_sig_time,
        new_sampling_rate,
        temp_time_before_peak,
        temp_time_after_peak,
    )

    (
        peak_time_from_corr,
        corr_heights,
        corr_ind,
    ) = extract_potential_peaks_from_correlation(
        corrs,
        corr_times,
        sampling_rate,
        corr_peak_extraction_method,
        fix_corr_peaks_by_height,
        fixpeaks_by_height_time_boundaries,
    )

    (
        peak_time_from_corr_height_filtered,
        peak_time_from_corr_rri_filtered,
    ) = handle_anomalies_in_peak_time_from_corr(
        peak_time_from_corr,
        sig_time,
        resampled_clean_sig,
        resampled_clean_sig_time,
        corrs,
        corr_ind,
        min_n_confident_peaks,
        corr_heights,
        thr_corr_height,
        relative_rri_for_temp_min,
        relative_rri_for_temp_max,
        min_bpm=min_bpm,
        max_bpm=max_bpm,
        max_time_after_last_peak=max_time_after_last_peak,
        use_rri_to_peak_time=use_rri_to_peak_time,
        find_anomalies_threshold=find_anomalies_threshold,
        move_average_rri_window=move_average_rri_window,
    )

    final_peak_time = fix_final_peaks(
        peak_time_from_corr_rri_filtered,
        orig_sig,
        orig_sig_time,
        orig_sampling_rate,
        corrs,
        corr_times,
        sampling_rate,
        fix_interpl_peaks_by_height,
        fix_added_interpl_peaks_by_height,
        fixpeaks_by_height_time_boundaries,
        k_nearest_intervals,
        n_nan_estimation_method,
        interpolate_args,
        min_bpm=min_bpm,
        max_bpm=max_bpm,
    )

    if output_format == "only_final":
        return final_peak_time
    else:
        return export_debug_info(
            resampled_clean_sig,
            resampled_clean_sig_time,
            new_sampling_rate,
            height_min,
            height_max,
            potential_peak_time_for_temp,
            peak_time_for_temp_confident,
            med_template,
            corrs,
            corr_times,
            peak_time_from_corr,
            peak_time_from_corr_height_filtered,
            peak_time_from_corr_rri_filtered,
            final_peak_time,
            debug_out_path=debug_out_path,
        )
