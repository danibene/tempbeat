from pathlib import Path
import pytest
from pytest import TempPathFactory

from tempbeat.skeleton import extract_peak_times_from_wav, main
from neurokit2.signal import signal_distort
from neurokit2 import data
import scipy
import numpy as np

__author__ = "danibene"
__copyright__ = "danibene"
__license__ = "MIT"

# PyTest fixture of wav file
@pytest.fixture
def wav_file(tmp_path_factory: TempPathFactory):
    """Create a temporary WAV file for testing"""
    sampling_rate = 100
    ecg_data = data("bio_resting_5min_100hz")
    clean_ecg = ecg_data["ECG"].values
    duration = len(clean_ecg) / sampling_rate

    pretend_audio = signal_distort(
        clean_ecg,
        sampling_rate=sampling_rate,
        noise_amplitude=0.5,
        artifacts_amplitude=1,
        artifacts_number=int(duration / 10),
        artifacts_frequency=2,
        random_state=42,
    )

    # Save to wav using pytest tmp_path fixture
    wav_path = Path(tmp_path_factory.mktemp("tempbeat"), "temp.wav")
    scipy.io.wavfile.write(wav_path, sampling_rate, pretend_audio)

    return wav_path

def test_extract_peak_times_from_wav(wav_file):
    """Test extract_peak_times_from_wav"""
    peak_time = extract_peak_times_from_wav(wav_file, method="temp")
    assert isinstance(peak_time, np.ndarray)
    assert len(peak_time) > 0
    assert isinstance(peak_time[0], float)


def test_main(capsys, wav_file):
    """CLI Tests"""
    # capsys is a pytest fixture that allows asserts against stdout/stderr
    # https://docs.pytest.org/en/stable/capture.html
    main([str(wav_file)])
    captured = capsys.readouterr()
    csv_file = str(wav_file).replace(".wav", "_peak_time.csv")
    assert f"Peak times saved to {csv_file}" in captured.out
