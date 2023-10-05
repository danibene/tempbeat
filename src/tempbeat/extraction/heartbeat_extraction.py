from pathlib import Path
from typing import Optional, Tuple, Union

import neurokit2 as nk
import numpy as np
import scipy

from ..utils.misc_utils import get_func_kwargs, write_dict_to_json
from ..utils.timestamps import (
    samp_to_timestamp,
    sampling_rate_to_sig_time,
    sig_time_to_sampling_rate,
    timestamp_to_samp,
)
from .anomaly_treatment import fix_final_peaks, remove_anomalies_from_corr_peaks
from .correlation import correlate_templates_with_signal
from .preprocessing_heartbeat import clean_and_resample_signal, fixpeaks_by_height
from .preprocessing_signal import drop_missing
from .template_generation import generate_template_from_signal


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
            Path(
                Path(debug_out_path).parent,
                Path(debug_out_path).stem,
                Path(debug_out_path).stem + time_str + ".json",
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
    relative_rri_for_temp_min: float = -2.5,
    relative_rri_for_temp_max: float = 2.5,
    min_n_confident_peaks: int = 20,
    max_time_after_last_peak: int = 5,
    clean_method: str = "own_filt",
    highcut: int = 25,
    fix_corr_peaks_by_height: bool = False,
    fix_interpl_peaks_by_height: bool = False,
    fix_added_interpl_peaks_by_height: bool = False,
    corr_peak_extraction_method: str = "nk_ecg_process",
    k_nearest_intervals: int = 8,
    n_nan_estimation_method: str = "round",
    interpolate_args: dict = {"method": "akima"},
    temp_time_before_peak: float = 0.3,
    temp_time_after_peak: float = 0.3,
    fixpeaks_by_height_time_boundaries: float = None,
    move_average_rri_window: int = 3,
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

    med_template, template_debug_out = generate_template_from_signal(
        resampled_clean_sig=resampled_clean_sig,
        resampled_clean_sig_time=resampled_clean_sig_time,
        sampling_rate=new_sampling_rate,
        min_bpm=min_bpm,
        max_bpm=max_bpm,
        min_n_peaks_for_temp_confident=min_n_peaks_for_temp_confident,
        relative_peak_height_for_temp_min=relative_peak_height_for_temp_min,
        relative_peak_height_for_temp_max=relative_peak_height_for_temp_max,
        temp_time_before_peak=temp_time_before_peak,
        temp_time_after_peak=temp_time_after_peak,
        relative_rri_for_temp_min=relative_rri_for_temp_min,
        relative_rri_for_temp_max=relative_rri_for_temp_max,
        move_average_rri_window=move_average_rri_window,
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
    ) = remove_anomalies_from_corr_peaks(
        peak_time_from_corr,
        sig_time,
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
            template_debug_out["height_min"],
            template_debug_out["height_max"],
            template_debug_out["potential_peak_time_for_temp"],
            template_debug_out["peak_time_for_temp_confident"],
            med_template,
            corrs,
            corr_times,
            peak_time_from_corr,
            peak_time_from_corr_height_filtered,
            peak_time_from_corr_rri_filtered,
            final_peak_time,
            debug_out_path=debug_out_path,
        )


def hb_extract(
    sig: np.ndarray,
    sampling_rate: int = 1000,
    sig_time: Optional[np.ndarray] = None,
    sig_name: Optional[str] = None,
    method: Optional[str] = None,
    subtract_mean: bool = True,
    hb_extract_algo_kwargs: Optional[dict] = None,
) -> np.ndarray:
    """
    Extract heartbeat times from a signal.

    Parameters
    ----------
    sig : np.ndarray
        The input signal.
    sampling_rate : int, optional
        The sampling rate of the signal.
    sig_time : Optional[np.ndarray], optional
        The time values corresponding to the signal.
    sig_name : Optional[str], optional
        The name of the signal.
    method : Optional[str], optional
        The method for extracting heartbeat times.
    subtract_mean : bool, optional
        Whether to subtract the mean from the signal.
    hb_extract_algo_kwargs : Optional[dict], optional
        Keyword arguments for the heartbeat extraction algorithm.

    Returns
    -------
    np.ndarray
        The extracted heartbeat times.
    """

    if hb_extract_algo_kwargs is None:
        hb_extract_algo_kwargs = {}
    if sig_time is None:
        sig_time = sampling_rate_to_sig_time(sig=sig, sampling_rate=sampling_rate)
    else:
        sampling_rate = sig_time_to_sampling_rate(sig_time=sig_time)
    sig, sig_time = drop_missing(sig, sig_time=sig_time)
    if subtract_mean:
        sig = sig - np.nanmean(sig)
    if method is None:
        if sig_name is None:
            method = "nk_neurokit"
        elif sig_name == "zephyr_ecg":
            method = "nk_neurokit"
        elif sig_name == "ti_ppg":
            method = "nk_elgendi"
        else:
            method = "matlab"
    if "temp" in method:
        return temp_hb_extract(
            sig=sig,
            sig_time=sig_time,
            sampling_rate=sampling_rate,
            **get_func_kwargs(temp_hb_extract, **hb_extract_algo_kwargs)
        )
    if "ecg_audio" in method.lower() or "nk" in method.lower():
        if "nk" in method:
            nk_method = method[3:]
            if sampling_rate > 1000:
                old_sampling_rate = sampling_rate
                sampling_rate = 1000
                sig, sig_time = scipy.signal.resample(
                    sig,
                    int(len(sig) * (sampling_rate / old_sampling_rate)),
                    t=sig_time,
                )

            if "nk_ecg_" in method or "nk_ppg_" in method.lower():
                nk_method = nk_method[4:]

            if "ppg" in str(sig_name) or "nk_ppg_" in method.lower():
                processed = nk.ppg.ppg_process(
                    sig, sampling_rate=sampling_rate, method=nk_method
                )
                bin_peak_col = "PPG_Peaks"
            else:
                processed = nk.ecg.ecg_process(
                    sig,
                    sampling_rate=sampling_rate,
                    method=nk_method,
                    **get_func_kwargs(
                        nk.ecg.ecg_process,
                        exclude_keys=["method"],
                        **hb_extract_algo_kwargs
                    )
                )
                bin_peak_col = "ECG_R_Peaks"
            bin_peak_sig = processed[0][bin_peak_col].values
        else:
            bin_peak_sig = sig
        peak_as_one = np.array(bin_peak_sig / np.nanmax(bin_peak_sig)).astype(int)
        peak_samp = np.where(peak_as_one == 1)[0]
        peak_time = samp_to_timestamp(
            samp=peak_samp, sampling_rate=sampling_rate, sig_time=sig_time
        )
        return peak_time
    else:
        return (
            get_mat_hb_extract(
                sig=sig,
                sampling_rate=sampling_rate,
                **get_func_kwargs(get_mat_hb_extract, **hb_extract_algo_kwargs)
            )
            + sig_time[0]
        )


def get_mat_hb_extract(
    sig: np.ndarray,
    sampling_rate: int = 1000,
    detector_func_name: str = "FilterHBDetection",
    code_path: Optional[Union[str, Path]] = None,
):
    if code_path is None:
        code_path = Path(__file__).parent.parent.parent / "matlab"

    m_file_paths = list(Path(code_path).rglob("*" + detector_func_name + ".m"))
    if len(m_file_paths) == 0:
        raise ValueError(
            detector_func_name + ".m not found in the code path: " + str(code_path)
        )
    else:
        exist_code_path_list = [p.parent for p in m_file_paths]

    try:
        import matlab.engine
    except ImportError:
        raise ImportError("Matlab engine is not installed.")

    eng = matlab.engine.start_matlab()

    for path in exist_code_path_list:
        s = eng.genpath(str(path))
        eng.addpath(s, nargout=0)

    # [bpm,timeVector,peakTime,peakHeight,filterEnveloppe,filterData]
    # = FilterHBDetection(inputAudio,fs,debug)
    eng.workspace["inputAudio"] = matlab.double(np.vstack(sig).astype(dtype="float64"))
    eng.workspace["fs"] = matlab.double(sampling_rate)
    eng.workspace["debug"] = matlab.double(0)
    string_eval = detector_func_name + "(inputAudio,fs,debug);"
    _, _, prediction, _, _, _ = eng.eval(string_eval, nargout=6)

    eng.quit()
    peak_time = np.hstack(np.asarray(prediction))
    return peak_time
