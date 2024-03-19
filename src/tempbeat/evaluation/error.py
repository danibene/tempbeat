from typing import Callable

import numpy as np


def get_error(
    a: np.ndarray,
    b: np.ndarray,
    subtract_cent: bool = False,
    cent_func: Callable = np.median,
) -> np.ndarray:
    """
    Get the error between two arrays.

    This function computes the error between two arrays. The error is defined as the difference between the
    two arrays. If `subtract_cent` is True, the median (by default) of the difference is subtracted from the difference.
    This is useful for computing the error between two arrays that are not aligned.

    Parameters
    ----------
    a : np.ndarray
        The first array.
    b : np.ndarray
        The second array.
    subtract_cent : bool, optional
        Whether to subtract the median of the difference from the difference. Default is False.
    cent_func : Callable, optional
        The function used to compute the central tendancy. Default is np.median.
    """
    # Compute the difference between the two arrays
    diff = np.subtract(a, b)

    # Subtract the central tendancy from the difference
    if subtract_cent:
        diff -= cent_func(diff)

    return diff


def get_mae(
    a: np.ndarray,
    b: np.ndarray,
    subtract_cent: bool = False,
    cent_func: Callable = np.median,
):
    """
    Get the mean absolute error between two arrays.

    This function computes the mean absolute error between two arrays. The mean absolute error is defined as the
    mean of the absolute difference between the two arrays. If `subtract_cent` is True, the median (by default)
    of the difference is subtracted from the difference. This is useful for computing the error between two arrays
    that are not aligned.

    Parameters
    ----------
    a : np.ndarray
        The first array.
    b : np.ndarray
        The second array.
    subtract_cent : bool, optional
        Whether to subtract the median of the difference from the difference. Default is False.
    cent_func : Callable, optional
        The function used to compute the central tendancy. Default is np.median.
    """
    # Compute the difference between the two arrays
    diff = get_error(a, b, subtract_cent=subtract_cent, cent_func=cent_func)

    # Compute the mean absolute error
    mae = np.mean(np.abs(diff))

    return mae
