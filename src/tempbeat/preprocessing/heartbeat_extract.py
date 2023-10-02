import pathlib

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
    relative_rri_min: float = -2.5,
    relative_rri_max: float = 2.5,
    absolute_diff_rri_min: float = -np.inf,
    absolute_diff_rri_max: float = np.inf,
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
    relative_rri_min : float, optional
        Minimum relative RRI.
    relative_rri_max : float, optional
        Maximum relative RRI.
    absolute_diff_rri_min : float, optional
        Minimum absolute difference in RRI.
    absolute_diff_rri_max : float, optional
        Maximum absolute difference in RRI.
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

    new_sampling_rate = 100
    div = sampling_rate / new_sampling_rate

    clean_sig_r, clean_sig_time_r = scipy.signal.resample(
        clean_sig, num=int(len(clean_sig) / div), t=sig_time
    )

    window_time = 60 / min_bpm
    window = int(new_sampling_rate * window_time) * 2
    potential_peaks = roll_func(clean_sig_r, window=window, func=np.max)
    stand_peaks = nk.standardize(potential_peaks, robust=True)
    peaks_no_outliers = potential_peaks[
        (stand_peaks <= relative_peak_height_for_temp_max)
        & (stand_peaks >= relative_peak_height_for_temp_min)
    ]
    if len(peaks_no_outliers) < min_n_peaks_for_temp_confident:
        peaks_no_outliers = potential_peaks[
            argtop_k(-1 * np.abs(stand_peaks), k=min_n_confident_peaks)
        ]
    height_min = np.min(peaks_no_outliers)
    height_max = np.max(peaks_no_outliers)
    peak_info = nk.signal_findpeaks(clean_sig_r)
    good_peaks = peak_info["Peaks"][
        (clean_sig_r[peak_info["Peaks"]] >= height_min)
        & (clean_sig_r[peak_info["Peaks"]] <= height_max)
    ]
    peak_time_for_temp = samp_to_timestamp(good_peaks, sig_time=clean_sig_time_r)

    rri, rri_time = peak_time_to_rri(
        peak_time_for_temp,
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

    if use_rri_to_peak_time:
        peak_time_for_temp_confident = rri_to_peak_time(rri=rri, rri_time=rri_time)
    else:
        peak_time_for_temp_confident = np.concatenate(
            (
                np.array([peak_time_for_temp[0]]),
                rri_time[(np.abs(nk.standardize(rri, robust=True)) < 2)],
            )
        )

    # idea: check for correlation of median template with some kind of saved
    # template (maybe derived from median over multiple participants)
    # to improve robustness
    templates = np.array([])

    for peak in peak_time_for_temp_confident:
        # TODO: make the window length based on samples rather than times
        # so that there aren't rounding errors
        hb_sig, hb_sig_time = get_local_hb_sig(
            peak=peak,
            sig=clean_sig_r,
            sig_time=clean_sig_time_r,
            sampling_rate=new_sampling_rate,
            time_before_peak=temp_time_before_peak,
            time_after_peak=temp_time_after_peak,
        )
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

    corrs = []
    corr_times = []
    sig = clean_sig_r
    sig_time = clean_sig_time_r
    sampling_rate = new_sampling_rate
    for time in sig_time:
        hb_sig, hb_sig_time = get_local_hb_sig(
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

    # can use the neurokit fixpeaks afterwards with the large missing values
    # max time_before_peak/time_after_peak in fix_local hb sig with clean can be based
    # on expected deviation from the linear interpolation i.e. mean
    # then do final fix_local on raw data to account for resampling to 100 Hz
    if corr_peak_extraction_method == "nk_ecg_peaks_nabian2018":
        _, processed_info = nk.ecg_peaks(
            corrs, method="nabian2018", sampling_rate=sampling_rate
        )
    elif corr_peak_extraction_method == "nk_ecg_process_kalidas2017":
        processed_signals, processed_info = nk.ecg_process(
            corrs, sampling_rate=sampling_rate, method="kalidas2017"
        )
    else:
        processed_signals, processed_info = nk.ecg_process(
            corrs, sampling_rate=sampling_rate
        )
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
    peak_time_from_corr_confident = peak_time_from_corr[
        (nk.standardize(corr_heights, robust=True) >= thr_corr_height)
    ]
    if len(peak_time_from_corr_confident) < min_n_confident_peaks:
        peak_time_from_corr_confident = peak_time_from_corr[
            argtop_k(corr_heights, k=min_n_confident_peaks)
        ]

    # here is where the peak fixing for the higher fs signal could go?
    if find_anomalies_threshold is None:
        rri, rri_time = peak_time_to_rri(
            peak_time_from_corr_confident,
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
            new_peak_time = rri_to_peak_time(rri=rri, rri_time=rri_time)
        else:
            new_peak_time = np.concatenate(
                (np.array([peak_time_from_corr_confident[0]]), rri_time)
            )
    else:
        anomalies_score = find_anomalies(
            peak_time_from_corr_confident,
            sig_info={"sig": clean_sig_r, "time": clean_sig_time_r},
        )
        anomalies = anomalies_score > find_anomalies_threshold
        if (
            len(peak_time_from_corr_confident)
            - len(peak_time_from_corr_confident[anomalies])
            < min_n_confident_peaks
        ):
            anomalies = argtop_k(
                anomalies_score,
                k=len(peak_time_from_corr_confident) - min_n_confident_peaks,
            )
        new_peak_time = peak_time_from_corr_confident.copy()
        new_peak_time[anomalies] = np.nan
        new_peak_time = new_peak_time[np.isfinite(new_peak_time)]

    min_last_peak_time = np.max(sig_time) - max_time_after_last_peak
    if np.max(new_peak_time) < min_last_peak_time:
        new_last_peak = peak_time_from_corr[
            argtop_k(corrs[ind][peak_time_from_corr > min_last_peak_time], k=1)
        ]
        new_peak_time = np.append(new_peak_time, new_last_peak)

    rri, rri_time = peak_time_to_rri(
        new_peak_time, min_rri=60000 / max_bpm, max_rri=60000 / min_bpm
    )

    new_peaks = timestamp_to_samp(new_peak_time, sig_time=sig_time)

    fixed_new_peaks = signal_fixpeaks(
        new_peaks,
        method="neurokit",
        sampling_rate=sampling_rate,
        interval_min=np.nanmin(rri) / 1000,
        interval_max=np.nanmax(rri) / 1000,
        # relative_interval_min=-2.5,
        # relative_interval_max=2.5,
        robust=True,
        k_nearest_intervals=k_nearest_intervals,
        n_nan_estimation_method=n_nan_estimation_method,
        interpolate_args=interpolate_args,
    )
    if fix_interpl_peaks_by_height:
        final_peak_time = fixpeaks_by_height(
            samp_to_timestamp(fixed_new_peaks, sig_time=sig_time),
            sig_info={
                "time": corr_times,
                "sig": corrs,
                "sampling_rate": sampling_rate,
            },
            time_boundaries=fixpeaks_by_height_time_boundaries,
        )
    elif fix_added_interpl_peaks_by_height:
        dec = 1
        fixed_peak_time = samp_to_timestamp(fixed_new_peaks, sig_time=sig_time)
        added_peak_time = np.array(
            [
                peak
                for peak in fixed_peak_time
                if np.round(peak, 1) not in np.round(new_peak_time, dec)
            ]
        )
        kept_peak_time = np.array(
            [
                peak
                for peak in fixed_peak_time
                if np.round(peak, 1) in np.round(new_peak_time, dec)
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
        final_peak_time = samp_to_timestamp(fixed_new_peaks, sig_time=sig_time)
    if output_format == "only_final":
        return final_peak_time
    else:
        debug_out = {}
        # debug_out["orig_sig"] = orig_sig
        # debug_out["orig_sig_time"] = orig_sig_time
        # debug_out["orig_sampling_rate"] = orig_sampling_rate
        debug_out["clean_sig_r"] = clean_sig_r
        debug_out["clean_sig_time_r"] = clean_sig_time_r
        debug_out["new_sampling_rate"] = new_sampling_rate
        debug_out["height_min"] = height_min
        debug_out["height_max"] = height_max
        debug_out["peak_time_for_temp"] = peak_time_for_temp
        debug_out["peak_time_for_temp_confident"] = peak_time_for_temp_confident
        debug_out["med_template"] = med_template
        debug_out["corrs"] = corrs
        debug_out["corr_times"] = corr_times
        debug_out["peak_time_from_corr"] = peak_time_from_corr
        debug_out["peak_time_from_corr_confident"] = peak_time_from_corr_confident
        debug_out["new_peak_time"] = new_peak_time
        debug_out["final_peak_time"] = final_peak_time
        if debug_out_path is None:
            return final_peak_time, debug_out
        else:
            time_str = "_" + str(int(np.min(clean_sig_time_r)))
            this_debug_out_path = str(
                pathlib.Path(
                    pathlib.Path(debug_out_path).parent,
                    pathlib.Path(debug_out_path).stem,
                    pathlib.Path(debug_out_path).stem + time_str + ".json",
                )
            )
            write_dict_to_json(debug_out, json_path=this_debug_out_path)
            return final_peak_time