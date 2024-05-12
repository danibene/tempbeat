#!/usr/bin/env python

import time
from pathlib import Path

import matplotlib.pyplot as plt

# import neurokit2 as nk
import numpy as np
import pandas as pd
import soundfile as sf

# from tempbeat.evaluation.compare_bpm import get_bpm_mae_from_peak_time
from tempbeat.extraction.heartbeat_extraction import hb_extract

# from tempbeat.extraction.interval_conversion import peak_time_to_rri
# from tempbeat.utils.matlab_utils import quit_matlab, set_matlab
# from tempbeat.utils.timestamp import (
#     sampling_rate_to_sig_time,
#     sig_time_to_sampling_rate,
# )


def read_audio_section(filename, start_time, stop_time):
    track = sf.SoundFile(filename)

    can_seek = track.seekable()  # True
    if not can_seek:
        raise ValueError("Not compatible with seeking")

    sr = track.samplerate
    start_frame = sr * start_time
    frames_to_read = sr * (stop_time - start_time)
    track.seek(int(start_frame))
    audio_section = track.read(int(frames_to_read))
    return audio_section, sr


def main() -> None:
    """Main analysis function"""
    # Set up the directories
    current_dir = Path(__file__).resolve().parent
    repo_dir = current_dir.parents[1]
    data_dir = repo_dir / "data"
    record_dir = data_dir / "DB20220227/SyncOriginal/03_danielle2"

    # Set up the audio file
    X_START = 900
    X_STOP = 1200
    audio_file = record_dir / "IEM_L.wav"
    audio_start_time = X_START
    audio_stop_time = X_STOP

    # Set up the ECG file
    # ecg_audio_file = record_dir / "ECG_audio.wav"
    ecg_file = record_dir / "ZephyrECG.csv"
    ecg_start_time = X_START
    ecg_stop_time = X_STOP

    XLIM = (210, 220)

    # Read the audio and ECG files
    audio_signal, audio_sr = read_audio_section(
        audio_file, audio_start_time, audio_stop_time
    )
    # audio_time = sampling_rate_to_sig_time(audio_signal, audio_sr)
    ecg = pd.read_csv(ecg_file)
    ecg.columns = ["t", "y"]
    ecg = ecg[(ecg["t"] >= ecg_start_time) & (ecg["t"] <= ecg_stop_time)]
    ecg_time = ecg["t"].to_numpy()
    ecg_time = ecg_time - ecg_time[0]
    ecg_signal = ecg["y"].to_numpy()
    # ecg_sr = sig_time_to_sampling_rate(ecg_time)

    """
    ecg_audio_signal, ecg_audio_sr = read_audio_section(
        ecg_audio_file, ecg_start_time, ecg_stop_time
    )
    ecg_audio_time = sampling_rate_to_sig_time(ecg_audio_signal, ecg_sr)

    _, ecg_peaks_dict = nk.ecg_process(ecg_signal, sampling_rate=ecg_sr)
    ecg_peaks = ecg_peaks_dict["ECG_R_Peaks"]
    """

    # Extract the heartbeats
    hb_extract_kwargs = {
        "fix_interpl_peaks_by_height": True,
        "max_time_after_last_peak": np.inf,
        "output_format": "full",
    }

    # Start timer
    start_time = time.time()
    audio_output = hb_extract(
        audio_signal, audio_sr, method="temp", hb_extract_algo_kwargs=hb_extract_kwargs
    )
    end_time = time.time()
    print(f"Elapsed time: {end_time - start_time}")
    # audio_peak_time = audio_output[0]
    print(audio_output[1].keys())

    resampled_clean_sig = audio_output[1]["resampled_clean_sig"]
    resampled_clean_sig_time = audio_output[1]["resampled_clean_sig_time"]
    """
    # Plot the resampled clean signal
    plt.plot(resampled_clean_sig_time, resampled_clean_sig)
    plt.xlim(XLIM)
    plt.show()

    template_height_min = audio_output[1]["template_height_min"]
    template_height_max = audio_output[1]["template_height_max"]

    # Plot the template height min and max as horizontal dashed lines over the resampled clean signal

    plt.plot(resampled_clean_sig_time, resampled_clean_sig)
    plt.axhline(template_height_min, color="red", linestyle="--")
    plt.axhline(template_height_max, color="red", linestyle="--")
    plt.xlim(XLIM)
    plt.show()

    potential_peak_time_for_temp = audio_output[1]["potential_peak_time_for_temp"]
    peak_time_for_temp_confident = audio_output[1]["peak_time_for_temp_confident"]

    # Plot the potential peak time for temp and peak time for temp confident
    # as vertical dashed lines over the resampled clean signal
    fig, ax = plt.subplots(2, 1, sharex=True)
    ax[0].plot(resampled_clean_sig_time, resampled_clean_sig)
    ax[0].vlines(potential_peak_time_for_temp, ymin=template_height_min, ymax=template_height_max, color="red", linestyle="--")

    ax[1].plot(resampled_clean_sig_time, resampled_clean_sig)
    ax[1].vlines(peak_time_for_temp_confident, ymin=template_height_min,
        ymax=template_height_max, color="green", linestyle="--")
    ax[1].set_xlim(XLIM)
    plt.show()

    plt.plot(resampled_clean_sig_time, resampled_clean_sig)
    plt.scatter(potential_peak_time_for_temp, np.repeat(template_height_min,
        len(potential_peak_time_for_temp)), color="red")
    plt.scatter(peak_time_for_temp_confident, np.repeat(template_height_max,
        len(peak_time_for_temp_confident)), color="green")
    plt.xlim(140, 160)
    plt.show()

    med_template = audio_output[1]["med_template"]

    # Plot the median template
    plt.plot(med_template)
    plt.title("Median template")
    plt.show() """

    corrs = audio_output[1]["corrs"]
    corr_times = audio_output[1]["corr_times"]

    # Plot the correlation values
    """
    plt.plot(corr_times, corrs)
    plt.title("Correlation values")
    plt.xlim(XLIM)
    plt.show()

    peak_time_from_corr = audio_output[1]["peak_time_from_corr"]
    peak_time_from_corr_height_filtered = audio_output[1][
        "peak_time_from_corr_height_filtered"
    ]
    """
    peak_time_from_corr_rri_filtered = audio_output[1][
        "peak_time_from_corr_rri_filtered"
    ]
    final_peak_time = audio_output[1]["final_peak_time"]

    seg_ecg_signal = ecg_signal[(ecg_time >= XLIM[0]) & (ecg_time <= XLIM[1])]
    seg_ecg_time = ecg_time[(ecg_time >= XLIM[0]) & (ecg_time <= XLIM[1])] - XLIM[0]
    seg_resampled_clean_sig = resampled_clean_sig[
        (resampled_clean_sig_time >= XLIM[0]) & (resampled_clean_sig_time <= XLIM[1])
    ]
    seg_resampled_clean_sig_time = (
        resampled_clean_sig_time[
            (resampled_clean_sig_time >= XLIM[0])
            & (resampled_clean_sig_time <= XLIM[1])
        ]
        - XLIM[0]
    )
    seg_corrs = corrs[(corr_times >= XLIM[0]) & (corr_times <= XLIM[1])]
    seg_corr_times = (
        corr_times[(corr_times >= XLIM[0]) & (corr_times <= XLIM[1])] - XLIM[0]
    )
    seg_peak_time_from_corr_rri_filtered = (
        peak_time_from_corr_rri_filtered[
            (peak_time_from_corr_rri_filtered >= XLIM[0])
            & (peak_time_from_corr_rri_filtered <= XLIM[1])
        ]
        - XLIM[0]
    )
    seg_final_peak_time = (
        final_peak_time[(final_peak_time >= XLIM[0]) & (final_peak_time <= XLIM[1])]
        - XLIM[0]
    )

    # Plot the peak time from corr, peak time from corr height filtered, peak time from corr rri filtered, and final peak time
    fig, ax = plt.subplots(4, 1, sharex=True)
    # Plot ECG and ECG peaks
    ax[0].plot(seg_ecg_time, seg_ecg_signal)
    ax[0].set_ylim([np.min(seg_ecg_signal), np.max(seg_ecg_signal)])
    # ax[0].scatter(ecg_time[ecg_peaks], ecg_signal[ecg_peaks], color="red")
    ax[0].set_title("ECG Peaks")
    # Plot resampled clean signal
    ax[1].plot(seg_resampled_clean_sig_time, seg_resampled_clean_sig)
    # ax[1].set_ylim([np.min(seg_resampled_clean_sig), np.max(seg_resampled_clean_sig)])
    ax[1].set_title("Resampled and filtered signal")
    # ax[1].set_title("Resampled and filtered signal with potential peak times for template")
    # ax[1].vlines(potential_peak_time_for_temp, ymin=template_height_min,
    #     ymax=template_height_max, color="red", linestyle="--")
    # Plot potential peak time for temp
    ax[2].plot(seg_corr_times, seg_corrs)
    ax[2].vlines(
        seg_peak_time_from_corr_rri_filtered,
        ymin=np.min(seg_corrs),
        ymax=np.max(seg_corrs),
        color="purple",
        linestyle="--",
    )
    ax[2].set_title(
        "Correlation of resampled and filtered signal with template: filtered peaks"
    )
    # ax[2].set_xlim(XLIM)
    # Plot peak time from corr
    ax[3].plot(seg_corr_times, seg_corrs)
    ax[3].vlines(
        seg_final_peak_time,
        ymin=np.min(seg_corrs),
        ymax=np.max(seg_corrs),
        color="green",
        linestyle="--",
    )
    ax[3].set_title(
        "Correlation of resampled and filtered signal with template: final peaks"
    )
    ax[3].set_xlabel("Time (seconds)")
    # ax[3].set_xlim(XLIM)
    plt.show()

    """
    # Plot the audio and ECG in subplots
    fig, ax = plt.subplots(2, 1, sharex=True)
    ax[0].plot(audio_time, audio_signal)
    ax[0].set_ylim(-0.025, 0.025)
    ax[0].set_title("Audio")
    ax[1].plot(ecg["t"], ecg["y"])
    ax[1].scatter(ecg_time[ecg_peaks], ecg_signal[ecg_peaks], color="red")
    ax[1].set_title("ECG")
    ax[1].set_xlim(50, 60)
    plt.show()
    """


if __name__ == "__main__":
    main()
