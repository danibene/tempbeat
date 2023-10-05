import inspect
import pathlib
import re
from typing import Any, Callable, Dict, List, Union
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
    pathlib.Path(json_path).resolve().parent.mkdir(parents=True, exist_ok=True)

    if not pathlib.Path(json_path).suffix == ".json":
        json_path = pathlib.Path(json_path + ".json")

    if not pathlib.Path(json_path).is_file() or rewrite:
        with open(str(json_path), "w") as json_file:
            dump(d, json_file, allow_nan=True, fmt=fmt)
    else:
        warn("Warning: " + str(json_path) + " already exists.")
