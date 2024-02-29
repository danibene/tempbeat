from typing import Tuple

import neurokit2 as nk
import numpy as np

from ..utils.misc_utils import a_moving_average, argtop_k, roll_func
from ..utils.timestamp import samp_to_timestamp
from .interval_conversion import peak_time_to_rri, rri_to_peak_time
from .segmentation import get_local_hb_sig


def extract_potential_peaks_for_template(
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
        The time values corresponding to the resampled cleaned signal.
    sampling_rate : int
        The sampling rate of the resampled cleaned signal.
    relative_peak_height_for_temp_min : float
        Minimum relative peak height for peaks used for computing template.
    relative_peak_height_for_temp_max : float
        Maximum relative peak height for peaks used for computing template.
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


def generate_template_from_peaks(
    peak_time_for_temp_confident: np.ndarray,
    resampled_clean_sig: np.ndarray,
    resampled_clean_sig_time: np.ndarray,
    sampling_rate: int,
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
    sampling_rate : int
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
            sampling_rate=sampling_rate,
            time_before_peak=temp_time_before_peak,
            time_after_peak=temp_time_after_peak,
        )
        # TODO: make the window length based on samples rather than times
        # so that there aren't rounding errors

        # Define the template length in terms of samples
        template_len = int(
            temp_time_before_peak * sampling_rate
            + temp_time_after_peak * sampling_rate
            - 2
        )

        if len(hb_sig) > template_len:
            if len(templates) < 1:
                templates = hb_sig[:template_len]
            else:
                templates = np.vstack((templates, hb_sig[:template_len]))

    med_template = np.nanmedian(np.array(templates), axis=0)

    return med_template


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
        Minimum relative RRI for intervals used to generate template.
    relative_rri_for_temp_max : float
        Maximum relative RRI for intervals used to generate template.
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


def generate_template_from_signal(
    resampled_clean_sig: np.ndarray,
    resampled_clean_sig_time: np.ndarray,
    sampling_rate: int,
    relative_peak_height_for_temp_min: float,
    relative_peak_height_for_temp_max: float,
    relative_rri_for_temp_min: float,
    relative_rri_for_temp_max: float,
    min_n_peaks_for_temp_confident: int,
    move_average_rri_window: int,
    min_bpm: int = 40,
    max_bpm: int = 200,
    temp_time_before_peak: float = 0.2,
    temp_time_after_peak: float = 0.2,
    use_rri_to_peak_time: bool = True,
) -> Tuple[np.ndarray, dict]:
    """
    Generate template based on cleaned signal.

    Parameters
    ----------
    resampled_clean_sig : np.ndarray
        Cleaned signal after resampling.
    resampled_clean_sig_time : np.ndarray
        Time values corresponding to the cleaned signal after resampling.
    sampling_rate : int
        The sampling rate of the resampled signal.
    relative_peak_height_for_temp_min : float
        Minimum relative peak height for peaks used to generate template.
    relative_peak_height_for_temp_max : float
        Maximum relative peak height for peaks used to generate template.
    relative_rri_for_temp_min : float
        Minimum relative RRI for intervals used to generate template.
    relative_rri_for_temp_max : float
        Maximum relative RRI for intervals used to generate template.
    min_n_peaks_for_temp_confident : int
        Minimum number of peaks to consider for generating template.
    move_average_rri_window : int
        Window size for moving average of RRI.
    min_bpm : int, optional
        The minimum heart rate in beats per minute.
    max_bpm : int, optional
        The maximum heart rate in beats per minute.
    temp_time_before_peak : float, optional
        Seconds before peak to take for segment used for template extraction.
    temp_time_after_peak : float, optional
        Seconds after peak to take for segment used for template extraction.
    use_rri_to_peak_time : bool, optional
        Whether to use RRI to peak time conversion. Default is True.

    Returns
    -------
    Tuple[np.ndarray, dict]
        Tuple containing the generated template and a dictionary containing
        information about the template generation.
    """
    # Extract potential peaks for template
    (
        potential_peak_time_for_temp,
        height_min,
        height_max,
    ) = extract_potential_peaks_for_template(
        resampled_clean_sig,
        resampled_clean_sig_time,
        sampling_rate,
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

    med_template = generate_template_from_peaks(
        peak_time_for_temp_confident,
        resampled_clean_sig,
        resampled_clean_sig_time,
        sampling_rate,
        temp_time_before_peak,
        temp_time_after_peak,
    )
    return med_template, {
        "height_min": height_min,
        "height_max": height_max,
        "peak_time_for_temp_confident": peak_time_for_temp_confident,
        "rri": rri,
        "rri_time": rri_time,
    }
