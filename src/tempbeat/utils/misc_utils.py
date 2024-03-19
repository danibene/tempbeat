import inspect
import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from warnings import warn

import numpy as np


def get_func_kwargs(
    func: Callable, exclude_keys: List[str] = [], **kwargs
) -> Dict[str, Any]:
    """
    Get keyword arguments relevant to a function.

    This function extracts keyword arguments that are relevant to the specified function. It uses
    the function's signature to identify valid keyword arguments.

    Parameters
    ----------
    func : Callable
        The target function.
    exclude_keys : List[str], optional
        List of keys to exclude from the extracted keyword arguments.
    **kwargs
        Additional keyword arguments.

    Returns
    -------
    Dict[str, Any]
        Dictionary containing keyword arguments relevant to the function.
    """
    # Retrieve the parameters of the target function
    func_args = list(inspect.signature(func).parameters)

    # Filter and extract keyword arguments
    func_kwargs = {
        k: kwargs.pop(k)
        for k in dict(kwargs)
        if k in func_args and k not in exclude_keys
    }

    return func_kwargs


def argtop_k(a: Union[List[Any], np.ndarray], k: int = 1, **kwargs) -> np.ndarray:
    """
    Return the indices of the top k elements in an array.

    This function returns the indices of the top k elements in the input array `a`.
    If `a` is a list, it is converted to a numpy array.

    Parameters
    ----------
    a : Union[List[Any], np.ndarray]
        The input array or list.
    k : int, optional
        The number of top elements to return. Default is 1.
    **kwargs
        Additional keyword arguments to be passed to the sorting function.

    Returns
    -------
    np.ndarray
        An array of indices corresponding to the top k elements in the input array.

    Notes
    -----
    See https://github.com/numpy/numpy/issues/15128
    """
    if type(a) is list:
        a = np.array(a)

    if k > len(a):
        k = len(a)

    return a.argsort()[-k:][::-1]


def top_k(
    a: Union[List[Any], np.ndarray], k: int = 1, **kwargs
) -> Union[List[Any], np.ndarray]:
    """
    Return the top k elements from the input array.

    Parameters
    ----------
    a : Union[List[Any], np.ndarray]
        The input array or list.
    k : int, optional
        The number of top elements to return. Default is 1.
    **kwargs
        Additional keyword arguments to be passed to the argtop_k function.

    Returns
    -------
    Union[List[Any], np.ndarray]
        An array or list containing the top k elements from the input array.

    Notes
    -----
    See https://github.com/numpy/numpy/issues/15128

    Examples
    --------
    >>> a = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5]
    >>> top_k(a, k=2)
    [9, 6]
    """
    return np.array(a)[argtop_k(a, k=k, **kwargs)]


def get_camel_case(s: str, first_upper: bool = False) -> str:
    """
    Convert a string to camelCase.

    This function takes a string and converts it to camelCase. It removes underscores and hyphens,
    and capitalizes the first letter of each word except for the first word if `first_upper` is set
    to False.

    Parameters
    ----------
    s : str
        The input string.
    first_upper : bool, optional
        Whether to capitalize the first letter of the resulting camelCase string. Default is False.

    Returns
    -------
    str
        The camelCase string.

    References
    ----------
    https://www.w3resource.com/python-exercises/string/python-data-type-string-exercise-96.php

    Examples
    --------
    >>> get_camel_case("hello_world")
    'helloWorld'

    >>> get_camel_case("hello_world", first_upper=True)
    'HelloWorld'
    """
    s = re.sub(r"(_|-)+", " ", s).title().replace(" ", "")
    if first_upper:
        return s
    return "".join([s[0].lower(), s[1:]])


def get_snake_case(s: str) -> str:
    """
    Convert a string to snake_case.

    This function takes a string and converts it to snake_case by inserting underscores
    before capital letters (except for the first letter).

    Parameters
    ----------
    s : str
        The input string.

    Returns
    -------
    str
        The snake_case string.

    Examples
    --------
    >>> get_snake_case("helloWorld")
    'hello_world'
    """
    return re.sub(r"(?<!^)(?=[A-Z])", "_", s).lower()


def write_dict_to_json(
    d: Dict[str, Any],
    json_path: str = "out.json",
    fmt: str = "%s",
    rewrite: bool = False,
) -> None:
    """
    Write a dictionary to a JSON file.

    Parameters
    ----------
    d : dict
        The dictionary to be written to the JSON file.
    json_path : str, optional
        The path to the JSON file. Defaults to "out.json".
    fmt : str, optional
        The format string used for formatting the JSON file. Defaults to "%s".
    rewrite : bool, optional
        If True, rewrite the file even if it already exists. Defaults to False.

    Raises
    ------
    ImportError
        If the 'json_tricks' module is not installed.
    """
    try:
        from json_tricks import dump
    except ImportError:
        raise ImportError(
            "Error in write_dict_to_json(): the 'json_tricks' module is required"
        )

    # if parent path does not exist create it
    Path(json_path).resolve().parent.mkdir(parents=True, exist_ok=True)

    if not Path(json_path).suffix == ".json":
        json_path = Path(json_path + ".json")

    if not Path(json_path).is_file() or rewrite:
        with open(str(json_path), "w") as json_file:
            dump(d, json_file, allow_nan=True, fmt=fmt)
    else:
        warn("Warning: " + str(json_path) + " already exists.")


def a_moving_average(y: np.ndarray, N: int = 5) -> np.ndarray:
    """
    Calculate the moving average of a 1-dimensional array.

    Parameters
    ----------
    y : np.ndarray
        Input array.
    N : int, optional
        Number of points used for the moving average window. Default is 5.

    Returns
    -------
    np.ndarray
        Smoothed array after applying the moving average.
    """
    y_padded = np.pad(y, (N // 2, N - 1 - N // 2), mode="edge")
    y_smooth = np.convolve(y_padded, np.ones((N,)) / N, mode="valid")
    return y_smooth


def roll_func(
    x: np.ndarray, window: int, func: callable, func_args: dict = {}
) -> np.ndarray:
    """
    Apply a rolling function to the input array.

    Parameters
    ----------
    x : np.ndarray
        Input array.
    window : int
        Size of the rolling window.
    func : callable
        Function to apply to each window.
    func_args : dict, optional
        Additional arguments to pass to the function.

    Returns
    -------
    np.ndarray
        Array resulting from applying the rolling function.
    """
    roll_x = np.array(
        [func(x[i : i + window], **func_args) for i in range(len(x) - window)]
    )
    return roll_x


def scale_and_clip_to_max_one(
    x: np.ndarray,
    min_value: float = 0,
    replace_min_value: float = 0,
    max_value: float = np.inf,
    replace_max_value: float = None,
    div_by_given_max: bool = True,
) -> np.ndarray:
    """
    Scale and clip values in an array to be between 0 and 1.

    Parameters
    ----------
    x : np.ndarray
        Input array.
    min_value : float, optional
        Minimum value to clip, by default 0.
    replace_min_value : float, optional
        Value to replace elements below `min_value`, by default 0.
    max_value : float, optional
        Maximum value to clip, by default np.inf.
    replace_max_value : float, optional
        Value to replace elements above `max_value`, by default None.
        If None, `max_value` is used.
    div_by_given_max : bool, optional
        If True, divide the array by `max_value`, by default True.
        If False, divide by the maximum value in the array.

    Returns
    -------
    np.ndarray
        Scaled and clipped array.
    """
    if replace_max_value is None:
        replace_max_value = max_value
    x[x < min_value] = replace_min_value
    x[x > max_value] = replace_max_value
    if div_by_given_max:
        return x / max_value
    else:
        return x / np.nanmax(x)


def drop_missing(
    sig: np.ndarray,
    sig_time: Optional[np.ndarray] = None,
    missing_value: float = np.nan,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Drop missing values from a signal.

    This function drops missing values from a signal. If a signal time array is provided, the
    function also drops the corresponding timestamps.

    Parameters
    ----------
    sig : np.ndarray
        Input signal.
    sig_time : Optional[np.ndarray], optional
        Array of timestamps corresponding to each sample. If not provided, timestamps
        are calculated based on the sampling rate.
    missing_value : float, optional
        Value to be considered missing. Default is np.nan.

    Returns
    -------
    Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]
        If a signal time array is provided, a tuple containing the signal and signal time arrays
        with missing values dropped. Otherwise, the signal array with missing values dropped.
    """
    if np.isnan(missing_value):
        not_missing = np.invert(np.isnan(sig))
    else:
        not_missing = np.where(sig == missing_value)
    sig = sig[not_missing]
    if sig_time is not None:
        sig_time = sig_time[not_missing]
        return sig, sig_time
    return sig


def export_debug_info(
    debug_out_path: Optional[str] = None, final_output_name="final_peak_time", **kwargs
):
    """
    Export debug information.

    Parameters
    ----------
    debug_out_path : str, optional
        Path to the debug output file. Default is None.
    """
    debug_out = kwargs
    final_output = kwargs[final_output_name]
    if debug_out_path is None:
        return final_output, debug_out
    else:
        this_debug_out_path = str(
            Path(
                Path(debug_out_path).parent,
                Path(debug_out_path).stem,
                Path(debug_out_path).stem + ".json",
            )
        )
        write_dict_to_json(debug_out, json_path=this_debug_out_path)
        return final_output
