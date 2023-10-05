from typing import Tuple, Union

import numpy as np
import scipy.signal

from .interpolation import interpolate_nonuniform
from .timestamp import sig_time_to_sampling_rate


def resample_nonuniform(
    sig: Union[np.ndarray, list],
    sig_time: Union[np.ndarray, list],
    new_sampling_rate: int = 1000,
    interpolate_method: str = "linear",
    use_matlab: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Resample a non-uniformly sampled signal to a new sampling rate.

    Parameters
    ----------
    sig : Union[np.ndarray, list]
        Input signal.
    sig_time : Union[np.ndarray, list]
        Array of timestamps corresponding to each sample.
    new_sampling_rate : int, optional
        The desired new sampling rate. Default is 1000 Hz.
    interpolate_method : str, optional
        Interpolation method for non-uniformly sampled signal. Default is "linear".
    use_matlab : bool, optional
        Whether to use MATLAB for resampling. Default is False.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Tuple containing the resampled signal and corresponding timestamps.

    Notes
    -----
    If `use_matlab` is True, the function uses MATLAB for resampling. Otherwise, it uses
    scipy's resample function.
    """
    if use_matlab:
        return _resample_matlab(sig, sig_time, new_sampling_rate)
    else:
        return _resample_scipy(sig, sig_time, new_sampling_rate, interpolate_method)


def _resample_matlab(
    sig: Union[np.ndarray, list],
    sig_time: Union[np.ndarray, list],
    new_sampling_rate: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Resample using MATLAB.

    Parameters
    ----------
    sig : Union[np.ndarray, list]
        Input signal.
    sig_time : Union[np.ndarray, list]
        Array of timestamps corresponding to each sample.
    new_sampling_rate : int
        The desired new sampling rate.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Tuple containing the resampled signal and corresponding timestamps.
    """
    try:
        import matlab.engine
    except ImportError:
        raise ImportError(
            "To use MATLAB for resampling, you must have MATLAB installed and the "
            "matlab.engine package installed in Python."
        )

    eng = matlab.engine.start_matlab()
    eng.workspace["x"] = matlab.double(np.vstack(sig).astype(dtype="float64"))
    eng.workspace["tx"] = matlab.double(np.vstack(sig_time).astype(dtype="float64"))
    eng.workspace["fs"] = matlab.double(new_sampling_rate)
    y, ty = eng.eval("resample(x, tx, fs);", nargout=2)
    new_sig = np.hstack(np.asarray(y))
    new_sig_time = np.hstack(np.asarray(ty))
    eng.quit()
    return new_sig, new_sig_time


def _resample_scipy(
    sig: Union[np.ndarray, list],
    sig_time: Union[np.ndarray, list],
    new_sampling_rate: int,
    interpolate_method: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Resample using scipy.

    Parameters
    ----------
    sig : Union[np.ndarray, list]
        Input signal.
    sig_time : Union[np.ndarray, list]
        Array of timestamps corresponding to each sample.
    new_sampling_rate : int
        The desired new sampling rate.
    interpolate_method : str
        Interpolation method for non-uniformly sampled signal.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Tuple containing the resampled signal and corresponding timestamps.
    """
    sampling_rate_interpl = sig_time_to_sampling_rate(
        sig_time, method="median", check_uniform=False
    )
    sig_interpl, sig_time_interpl = interpolate_nonuniform(
        sig,
        sig_time,
        sampling_rate=sampling_rate_interpl,
        method=interpolate_method,
    )
    new_n_samples = int(
        np.round(len(sig_time_interpl) * (new_sampling_rate / sampling_rate_interpl))
    )
    new_sig, new_sig_time = scipy.signal.resample(
        sig_interpl, new_n_samples, t=sig_time_interpl
    )
    return new_sig, new_sig_time
