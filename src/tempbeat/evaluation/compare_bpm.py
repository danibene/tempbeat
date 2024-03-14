from typing import Callable

import numpy as np

from ..extraction.interval_conversion import peak_time_to_rri
from ..utils.interpolation import interpolate_to_same_x
from .error import get_mae


def get_bpm_mae_from_rri(
    rri_a: np.ndarray,
    rri_b: np.ndarray,
    rri_time_a: np.ndarray,
    rri_time_b: np.ndarray,
    interpolation_rate: int = 2,
    subtract_cent: bool = False,
    cent_func: Callable = np.median,
    unit: str = "bpm",
    percentage: bool = False,
) -> float:
    """
    Get the mean absolute error of BPM given two RRI arrays.

    This function computes the mean absolute error between beats per minute (BPM) computed with two RRI arrays.

    Parameters
    ----------
    rri_a : np.ndarray
        The first RRI array.
    rri_b : np.ndarray
        The second RRI array.
    rri_time_a : np.ndarray
        The time array corresponding to the first RRI array.
    rri_time_b : np.ndarray
        The time array corresponding to the second RRI array.
    interpolation_rate : int, optional
        The interpolation rate. Default is 2.
    subtract_cent : bool, optional
        Whether to subtract the median of the difference from the difference. Default is False.
    cent_func : Callable, optional
        The function used to compute the central tendancy. Default is np.median.
    unit : str, optional
        The unit of the result. Default is "bpm". Other options are "rri".
    percentage : bool, optional
        Whether to return the result as a percentage. Default is False.
    """

    # Interpolate the RRI arrays to the same x values
    _, rri_a, rri_b = interpolate_to_same_x(
        a_x=rri_time_a,
        a_y=rri_a,
        b_x=rri_time_b,
        b_y=rri_b,
        interpolation_rate=interpolation_rate,
    )

    # Convert the RRI arrays to BPM arrays
    if unit == "bpm":
        rri_a = 60000 / rri_a
        rri_b = 60000 / rri_b

    # Compute the mean absolute error of the BPM
    mae = get_mae(rri_a, rri_b, subtract_cent=subtract_cent, cent_func=cent_func)

    if percentage:
        mae = mae / np.nanmean(rri_a) * 100

    return mae


def get_bpm_mae_from_peak_time(
    peak_time_a: np.ndarray,
    peak_time_b: np.ndarray,
    min_bpm: int = 40,
    max_bpm: int = 200,
    interpolation_rate: int = 2,
    subtract_cent: bool = False,
    cent_func: Callable = np.median,
    unit: str = "bpm",
    percentage: bool = False,
) -> float:
    """
    Get the mean absolute error of BPM given two peak time arrays.

    This function computes the mean absolute error between beats per minute (BPM) computed with two peak time arrays.

    Parameters
    ----------
    peak_time_a : np.ndarray
        The first peak time array.
    peak_time_b : np.ndarray
        The second peak time array.
    min_bpm : int, optional
        The minimum BPM considered as valid when converting peak times to R-R intervals. Default is 40.
    max_bpm : int, optional
        The maximum BPM considered as valid when converting peak times to R-R intervals. Default is 200.
    interpolation_rate : int, optional
        The interpolation rate. Default is 2.
    subtract_cent : bool, optional
        Whether to subtract the median of the difference from the difference. Default is False.
    cent_func : Callable, optional
        The function used to compute the central tendancy. Default is np.median.
    unit : str, optional
        The unit of the result. Default is "bpm". Other options are "rri".
    """
    # Convert the peak time arrays to RRI arrays
    if min_bpm is None:
        min_rri = None
    else:
        min_rri = 60000 / max_bpm
    if max_bpm is None:
        max_rri = None
    else:
        max_rri = 60000 / min_bpm
    rri_a, rri_time_a = peak_time_to_rri(peak_time_a, min_rri=min_rri, max_rri=max_rri)
    rri_b, rri_time_b = peak_time_to_rri(peak_time_b, min_rri=min_rri, max_rri=max_rri)

    # Compute the mean absolute error of the BPM
    mae = get_bpm_mae_from_rri(
        rri_a,
        rri_b,
        rri_time_a,
        rri_time_b,
        interpolation_rate=interpolation_rate,
        subtract_cent=subtract_cent,
        cent_func=cent_func,
        unit=unit,
        percentage = percentage
    )

    

    return mae
