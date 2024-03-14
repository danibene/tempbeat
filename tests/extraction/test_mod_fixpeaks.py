import neurokit2 as nk
import numpy as np
import numpy.random
import pytest

from tempbeat.extraction.mod_fixpeaks import signal_fixpeaks


@pytest.fixture
def testpeaks_for_neurokit_method():
    signal = nk.signal_simulate(duration=20, sampling_rate=1000, frequency=1)
    peaks_true = nk.signal_findpeaks(signal)["Peaks"]
    peaks = np.delete(peaks_true, [5, 6, 7, 8, 9, 10, 15, 16, 17, 19])  # create gaps
    # (I added more than in the example in the function docstring)
    peaks = np.sort(np.append(peaks, [1350, 11350, 18350]))  # add artifacts
    return peaks


@pytest.mark.parametrize("interval_max", [None, 1.5, 2.0])
def test_neurokit_method_returns_only_positive_indices(
    testpeaks_for_neurokit_method, interval_max
):
    peaks_corrected = signal_fixpeaks(
        peaks=testpeaks_for_neurokit_method,
        interval_min=0.5,
        interval_max=interval_max,
        method="neurokit",
    )
    assert np.all(peaks_corrected >= 0)


@pytest.mark.parametrize("interval_max", [None, 1.5, 2.0])
def test_neurokit_method_returns_no_duplicates(
    testpeaks_for_neurokit_method, interval_max
):
    peaks_corrected = signal_fixpeaks(
        peaks=testpeaks_for_neurokit_method,
        interval_min=0.5,
        interval_max=interval_max,
        method="neurokit",
    )
    assert np.unique(peaks_corrected).size == peaks_corrected.size


@pytest.mark.parametrize("interval_max", [None, 1.5, 2.0])
def test_neurokit_method_returns_strictly_increasing_indices(
    testpeaks_for_neurokit_method, interval_max
):
    peaks_corrected = signal_fixpeaks(
        peaks=testpeaks_for_neurokit_method,
        interval_min=0.5,
        interval_max=interval_max,
        method="neurokit",
    )
    assert np.all(np.diff(peaks_corrected) > 0)
