#!/usr/bin/env python

from neurokit2 import data, ecg_process, signal_distort

from tempbeat.evaluation.compare_bpm import get_bpm_mae_from_peak_time
from tempbeat.extraction.heartbeat_extraction import hb_extract
from tempbeat.utils.timestamp import samp_to_timestamp, sampling_rate_to_sig_time


def main() -> None:
    """Main analysis function"""
    # This is how you can use Neurokit2 to process ECG
    random_state = 42
    sampling_rate = 100
    ecg_data = data("bio_resting_5min_100hz")
    clean_ecg = ecg_data["ECG"].values
    duration = len(clean_ecg) / sampling_rate
    _, clean_rpeaks = ecg_process(clean_ecg, sampling_rate=sampling_rate)
    clean_peak_time = samp_to_timestamp(
        clean_rpeaks["ECG_R_Peaks"], sampling_rate=sampling_rate
    )

    for hb_extract_method in ["temp", "matlab"]:
        # Replace this with audio data
        pretend_audio_sig = signal_distort(
            clean_ecg,
            sampling_rate=sampling_rate,
            noise_amplitude=0.5,
            noise_frequency=10,
            artifacts_amplitude=1,
            artifacts_number=int(duration / 10),
            artifacts_frequency=2,
            random_state=random_state,
        )
        pretend_audio_sig_time = sampling_rate_to_sig_time(
            pretend_audio_sig, sampling_rate=sampling_rate
        )
        pretend_audio_peak_time = hb_extract(
            pretend_audio_sig,
            sig_time=pretend_audio_sig_time,
            sampling_rate=sampling_rate,
            method=hb_extract_method,
        )
        mae_clean_distorted = get_bpm_mae_from_peak_time(
            peak_time_a=clean_peak_time, peak_time_b=pretend_audio_peak_time
        )
        print(hb_extract_method)
        print(mae_clean_distorted)


if __name__ == "__main__":
    main()
