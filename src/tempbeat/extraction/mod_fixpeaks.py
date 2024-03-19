from typing import Optional
from warnings import warn

import numpy as np
import pandas as pd
from neurokit2.misc import NeuroKitWarning
from neurokit2.signal.signal_formatpeaks import _signal_formatpeaks_sanitize
from neurokit2.signal.signal_period import signal_period
from neurokit2.stats import standardize

from tempbeat.utils.misc_utils import argtop_k


def signal_fixpeaks(
    peaks: np.ndarray,
    sampling_rate: int = 1000,
    interval_min: Optional[float] = None,
    interval_max: Optional[float] = None,
    relative_interval_min: Optional[float] = None,
    relative_interval_max: Optional[float] = None,
    robust: bool = False,
    method: str = "neurokit",
    k_nearest_intervals: Optional[int] = None,
    n_nan_estimation_method: str = "floor",
    interpolate_args: dict = {"method": "linear"},
    **kwargs,
):
    """**Correct Erroneous Peak Placements**
    Identify and correct erroneous peak placements based on outliers in peak-to-peak differences
    (period).
    Parameters
    ----------
    peaks : list or array or DataFrame or Series or dict
        The samples at which the peaks occur. If an array is passed in, it is assumed that it was
        obtained with :func:`.signal_findpeaks`. If a DataFrame is passed in, it is assumed to be
        obtained with :func:`.ecg_findpeaks` or :func:`.ppg_findpeaks` and to be of the same length
        as the input signal.
    sampling_rate : int
        The sampling frequency of the signal that contains the peaks (in Hz, i.e., samples/second).
    interval_min : float
        Only when ``method = "neurokit"``. The minimum interval between the peaks.
    interval_max : float
        Only when ``method = "neurokit"``. The maximum interval between the peaks.
    relative_interval_min : float
        Only when ``method = "neurokit"``. The minimum interval between the peaks as relative to
        the sample (expressed in standard deviation from the mean).
    relative_interval_max : float
        Only when ``method = "neurokit"``. The maximum interval between the peaks as relative to
        the sample (expressed in standard deviation from the mean).
    robust : bool
        Only when ``method = "neurokit"``. Use a robust method of standardization (see
        :func:`.standardize`) for the relative thresholds.
    method : str
        Only ``"neurokit"``.
    **kwargs
        Other keyword arguments.
    Returns
    -------
    peaks_clean : array
        The corrected peak locations.
    artifacts : dict
        Only if ``method="Kubios"``. A dictionary containing the indices of artifacts, accessible
        with the keys ``"ectopic"``, ``"missed"``, ``"extra"``, and ``"longshort"``.
    See Also
    --------
    signal_findpeaks, ecg_findpeaks, ecg_peaks, ppg_findpeaks, ppg_peaks
    Examples
    --------
    .. ipython:: python
      import neurokit2 as nk
      # Simulate ECG data
      ecg = nk.ecg_simulate(duration=240, noise=0.25, heart_rate=70, random_state=42)
      # Identify and Correct Peaks using "Kubios" Method
      rpeaks_uncorrected = nk.ecg_findpeaks(ecg)
      @savefig p_signal_fixpeaks1.png scale=100%
      artifacts, rpeaks_corrected = nk.signal_fixpeaks(
          rpeaks_uncorrected, iterative=True, method="Kubios", show=True
      )
      @suppress
      plt.close()
    .. ipython:: python
      # Visualize Artifact Correction
      rate_corrected = nk.signal_rate(rpeaks_corrected, desired_length=len(ecg))
      rate_uncorrected = nk.signal_rate(rpeaks_uncorrected, desired_length=len(ecg))
      @savefig p_signal_fixpeaks2.png scale=100%
      nk.signal_plot(
          [rate_uncorrected, rate_corrected],
          labels=["Heart Rate Uncorrected", "Heart Rate Corrected"]
      )
      @suppress
      plt.close()
    .. ipython:: python
      import numpy as np
      # Simulate Abnormal Signals
      signal = nk.signal_simulate(duration=4, sampling_rate=1000, frequency=1)
      peaks_true = nk.signal_findpeaks(signal)["Peaks"]
      peaks = np.delete(peaks_true, [1])  # create gaps due to missing peaks
      signal = nk.signal_simulate(duration=20, sampling_rate=1000, frequency=1)
      peaks_true = nk.signal_findpeaks(signal)["Peaks"]
      peaks = np.delete(peaks_true, [5, 15])  # create gaps
      peaks = np.sort(np.append(peaks, [1350, 11350, 18350]))  # add artifacts
      # Identify and Correct Peaks using 'NeuroKit' Method
      peaks_corrected = nk.signal_fixpeaks(
          peaks=peaks, interval_min=0.5, interval_max=1.5, method="neurokit"
      )
      # Plot and shift original peaks to the right to see the difference.
      @savefig p_signal_fixpeaks3.png scale=100%
      nk.events_plot([peaks + 50, peaks_corrected], signal)
      @suppress
      plt.close()
    References
    ----------
    * Lipponen, J. A., & Tarvainen, M. P. (2019). A robust algorithm for heart rate variability time
      series artefact correction using novel beat classification. Journal of medical engineering &
      technology, 43(3), 173-181. 10.1080/03091902.2019.1640306
    """
    # Format input
    peaks = _signal_formatpeaks_sanitize(peaks)

    # Confirm that method is NeuroKit
    if method.lower() != "neurokit":
        raise ValueError("NeuroKit method is the only method available at the moment.")

    # Else method is NeuroKit
    return _signal_fixpeaks_neurokit(
        peaks,
        sampling_rate=sampling_rate,
        interval_min=interval_min,
        interval_max=interval_max,
        relative_interval_min=relative_interval_min,
        relative_interval_max=relative_interval_max,
        robust=robust,
        k_nearest_intervals=k_nearest_intervals,
        n_nan_estimation_method=n_nan_estimation_method,
        interpolate_args=interpolate_args,
    )


# =============================================================================
# Methods
# =============================================================================
def _signal_fixpeaks_neurokit(
    peaks: np.ndarray,
    sampling_rate: int = 1000,
    interval_min: Optional[float] = None,
    interval_max: Optional[float] = None,
    relative_interval_min: Optional[float] = None,
    relative_interval_max: Optional[float] = None,
    robust: bool = False,
    k_nearest_intervals: Optional[int] = None,
    n_nan_estimation_method: str = "floor",
    interpolate_args: dict = {"method": "linear"},
):
    """NeuroKit method."""

    peaks_clean = _remove_small(
        peaks, sampling_rate, interval_min, relative_interval_min, robust
    )
    peaks_clean = _interpolate_big(
        peaks_clean,
        sampling_rate,
        interval_max,
        relative_interval_max,
        robust,
        k_nearest_intervals=k_nearest_intervals,
        n_nan_estimation_method=n_nan_estimation_method,
        interpolate_args=interpolate_args,
    )

    valid_peaks = peaks_clean[peaks_clean >= 0]
    n_invalid_idcs = len(peaks_clean) - len(valid_peaks)
    if n_invalid_idcs > 0:
        warn(
            f" Negative peak indices detected in output. "
            f" Removing {n_invalid_idcs} invalid peaks. ",
            category=NeuroKitWarning,
        )
        peaks_clean = valid_peaks
    return peaks_clean


# =============================================================================
# NeuroKit
# =============================================================================
def _remove_small(
    peaks: np.ndarray,
    sampling_rate: int = 1000,
    interval_min: Optional[float] = None,
    relative_interval_min: Optional[float] = None,
    robust: bool = False,
):
    if interval_min is None and relative_interval_min is None:
        return peaks
    if interval_min is not None:
        interval = signal_period(
            peaks, sampling_rate=sampling_rate, desired_length=None
        )
        peaks = peaks[interval > interval_min]
    if relative_interval_min is not None:
        interval = signal_period(
            peaks, sampling_rate=sampling_rate, desired_length=None
        )
        peaks = peaks[standardize(interval, robust=robust) > relative_interval_min]
    return peaks


def _interpolate_big(
    peaks: np.ndarray,
    sampling_rate: int = 1000,
    interval_max: Optional[float] = None,
    relative_interval_max: Optional[float] = None,
    robust: bool = False,
    k_nearest_intervals: Optional[int] = None,
    n_nan_estimation_method: str = "floor",
    interpolate_args: dict = {"method": "linear"},
):
    if interval_max is None and relative_interval_max is None:
        return peaks
    if interval_max is not None:
        interval = signal_period(
            peaks, sampling_rate=sampling_rate, desired_length=None
        )
        peaks = _interpolate_missing(
            peaks=peaks,
            interval=interval,
            interval_max=interval_max,
            k_nearest_intervals=k_nearest_intervals,
            n_nan_estimation_method=n_nan_estimation_method,
            interpolate_args=interpolate_args,
        )
    if relative_interval_max is not None:
        interval = signal_period(
            peaks, sampling_rate=sampling_rate, desired_length=None
        )
        interval = standardize(interval, robust=robust)
        peaks = _interpolate_missing(
            peaks=peaks,
            interval=interval,
            interval_max=relative_interval_max,
            k_nearest_intervals=k_nearest_intervals,
            n_nan_estimation_method=n_nan_estimation_method,
            interpolate_args=interpolate_args,
        )
    return peaks


def _interpolate_missing(
    peaks: np.ndarray,
    interval: np.ndarray,
    interval_max: float,
    k_nearest_intervals: Optional[int] = None,
    n_nan_estimation_method: str = "floor",
    interpolate_args: dict = {"method": "linear"},
):
    outliers = interval > interval_max
    outliers_loc = np.where(outliers)[0]

    # interval returned by signal_period at index 0 is the mean of the intervals
    # so it does not actually correspond to whether the first peak is an outlier
    outliers_loc = outliers_loc[outliers_loc != 0]

    if np.sum(outliers) == 0:
        return peaks
    peaks_to_correct = peaks.copy().astype(float)

    interval_without_outliers = interval[np.invert(outliers)]
    mean_interval = np.nanmean(interval_without_outliers)

    # go through the outliers starting with the highest indices
    # so that the indices of the other outliers are not moved when
    # unknown intervas are inserted
    if k_nearest_intervals is not None:
        non_outlier_locs = np.arange(len(interval))[np.invert(outliers)]
    for loc in np.flip(outliers_loc):
        if k_nearest_intervals is not None:
            mean_interval = np.nanmean(
                interval[
                    non_outlier_locs[
                        argtop_k(
                            -1 * np.abs(non_outlier_locs - loc), k=k_nearest_intervals
                        )
                    ]
                ]
            )

        # compute number of NaNs to insert based on the mean interval
        if n_nan_estimation_method == "floor":
            n_nan = int(interval[loc] / mean_interval)
        else:
            n_nan = int(np.round(interval[loc] / mean_interval))

        # Delete peak corresponding to large interval and replace by N NaNs
        peaks_to_correct[loc] = np.nan
        peaks_to_correct = np.insert(peaks_to_correct, loc, [np.nan] * (n_nan - 1))
    # Interpolate values
    interpolated_peaks = (
        pd.Series(peaks_to_correct)
        .interpolate(limit_area="inside", **interpolate_args)
        .values
    )
    # If there are missing values remaining, remove
    peaks = interpolated_peaks[np.invert(np.isnan(interpolated_peaks))].astype(
        peaks.dtype
    )
    return peaks
