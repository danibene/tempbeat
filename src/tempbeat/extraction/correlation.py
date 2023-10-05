from typing import Tuple

import numpy as np

from .template_generation import get_local_hb_sig


def norm_corr(a: np.ndarray, b: np.ndarray, maxlags: int = 0) -> np.ndarray:
    """
    Calculate normalized cross-correlation between two 1-dimensional arrays.

    Parameters
    ----------
    a : np.ndarray
        First array.
    b : np.ndarray
        Second array.
    maxlags : int, optional
        Maximum lag to calculate. Default is 0.

    Returns
    -------
    np.ndarray
        Array containing the normalized cross-correlation.

    References
    ----------
    https://stackoverflow.com/questions/53436231/normalized-cross-correlation-in-python
    """
    Nx = len(a)

    if Nx != len(b):
        raise ValueError("a and b must be equal length")

    if maxlags is None:
        maxlags = Nx - 1

    if maxlags >= Nx or maxlags < 0:
        raise ValueError("maxlags must be None or strictly positive < %d" % Nx)

    a = (a - np.mean(a)) / (np.std(a) * len(a))
    b = (b - np.mean(b)) / (np.std(b))
    c = np.correlate(a, b, "full")
    c = c[Nx - 1 - maxlags : Nx + maxlags]

    return c


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
