import inspect
import re
from typing import Any, Callable, Dict, List

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


def argtop_k(a, k=1, **kwargs):
    # See https://github.com/numpy/numpy/issues/15128
    # return np.argpartition(a, -k, **kwargs)[-k:]
    if type(a) is list:
        a = np.array(a)
    if k > len(a):
        k = len(a)
    return a.argsort()[-k:][::-1]


def top_k(a, k=1, **kwargs):
    # See https://github.com/numpy/numpy/issues/15128
    return a[argtop_k(a, k=k, **kwargs)]


def get_camel_case(s, first_upper=False):
    # https://www.w3resource.com/python-exercises/string/python-data-type-string-exercise-96.php
    s = re.sub(r"(_|-)+", " ", s).title().replace(" ", "")
    if first_upper:
        return s
    return "".join([s[0].lower(), s[1:]])


def get_snake_case(s):
    return re.sub(r"(?<!^)(?=[A-Z])", "_", s).lower()
