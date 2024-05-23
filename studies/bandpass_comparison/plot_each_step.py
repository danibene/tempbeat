#!/usr/bin/env python

import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import soundfile as sf

from tempbeat.extraction.heartbeat_extraction import hb_extract


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
    export_dir_root_parent = Path(data_dir, "outputs", "corrplots")

    # Set up the audio file
    X_START = 900
    X_STOP = 1200
    audio_file = record_dir / "IEM_L.wav"
    audio_start_time = X_START
    audio_stop_time = X_STOP

    # Set up the ECG file
    ecg_file = record_dir / "ZephyrECG.csv"
    ecg_start_time = X_START
    ecg_stop_time = X_STOP

    XLIM = (210, 220)

    # Read the audio and ECG files
    audio_signal, audio_sr = read_audio_section(
        audio_file, audio_start_time, audio_stop_time
    )
    ecg = pd.read_csv(ecg_file)
    ecg.columns = ["t", "y"]
    ecg = ecg[(ecg["t"] >= ecg_start_time) & (ecg["t"] <= ecg_stop_time)]
    ecg_time = ecg["t"].to_numpy()
    ecg_time = ecg_time - ecg_time[0]
    ecg_signal = ecg["y"].to_numpy()

    args_dict = {
        "original": {
            "fix_interpl_peaks_by_height": False,
            "max_time_after_last_peak": 5,
            "output_format": "full",
        },
        "ignoretimeafterlastpeak": {
            "fix_interpl_peaks_by_height": False,
            "max_time_after_last_peak": np.inf,
            "output_format": "full",
        },
        "heightfix": {
            "fix_interpl_peaks_by_height": True,
            "max_time_after_last_peak": 5,
            "output_format": "full",
        },
        "heightfixignoretimeafterlastpeak": {
            "fix_interpl_peaks_by_height": True,
            "max_time_after_last_peak": np.inf,
            "output_format": "full",
        },
    }

    for key, args in args_dict.items():
        export_dir_root = export_dir_root_parent / key
        if not export_dir_root.exists():
            export_dir_root.mkdir(parents=True)

        hb_extract_kwargs = args
        # Start timer
        start_time = time.time()
        audio_output = hb_extract(
            audio_signal,
            audio_sr,
            method="temp",
            hb_extract_algo_kwargs=hb_extract_kwargs,
        )
        end_time = time.time()
        print(f"Elapsed time: {end_time - start_time}")
        print(audio_output[1].keys())

        resampled_clean_sig = audio_output[1]["resampled_clean_sig"]
        resampled_clean_sig_time = audio_output[1]["resampled_clean_sig_time"]

        corrs = audio_output[1]["corrs"]
        corr_times = audio_output[1]["corr_times"]

        peak_time_from_corr = audio_output[1]["peak_time_from_corr"]
        peak_time_from_corr_rri_filtered = audio_output[1][
            "peak_time_from_corr_rri_filtered"
        ]
        final_peak_time = audio_output[1]["final_peak_time"]

        seg_ecg_signal = ecg_signal[(ecg_time >= XLIM[0]) & (ecg_time <= XLIM[1])]
        seg_ecg_time = ecg_time[(ecg_time >= XLIM[0]) & (ecg_time <= XLIM[1])] - XLIM[0]
        seg_resampled_clean_sig = resampled_clean_sig[
            (resampled_clean_sig_time >= XLIM[0])
            & (resampled_clean_sig_time <= XLIM[1])
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

        seg_peak_time_from_corr = (
            peak_time_from_corr[
                (peak_time_from_corr >= XLIM[0]) & (peak_time_from_corr <= XLIM[1])
            ]
            - XLIM[0]
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

        # Plot the correlation of the pre-processed IEM and the template
        fig, ax = plt.subplots(3, 1, sharex=True, figsize=(5, 3))
        # Plot ECG and ECG peaks
        ax[0].plot(seg_ecg_time, seg_ecg_signal)
        ax[0].set_ylim([np.min(seg_ecg_signal), np.max(seg_ecg_signal)])
        ax[0].set_title("ECG")
        # Plot resampled clean signal
        ax[1].plot(seg_resampled_clean_sig_time, seg_resampled_clean_sig)
        ax[1].set_title("Pre-processed IEM")
        ax[2].plot(seg_corr_times, seg_corrs)
        ax[2].set_title("Correlation of pre-processed IEM and template")
        ax[2].set_xlabel("Time (seconds)")

        plt.tight_layout()
        plt.savefig(export_dir_root / "ecg_and_corr.pdf")

        # Plot the peak time from corr, peak time from corr height filtered,
        # peak time from corr rri filtered, and final peak time
        fig, ax = plt.subplots(5, 1, sharex=True, figsize=(5, 5))
        # Plot ECG and ECG peaks
        ax[0].plot(seg_ecg_time, seg_ecg_signal, alpha=0.75)
        ax[0].set_ylim([np.min(seg_ecg_signal), np.max(seg_ecg_signal)])
        ax[0].set_title("ECG")
        # Plot resampled clean signal
        ax[1].plot(seg_resampled_clean_sig_time, seg_resampled_clean_sig, alpha=0.75)
        ax[1].set_title("Pre-processed IEM")
        ax[2].plot(seg_corr_times, seg_corrs, alpha=0.75)
        ax[2].vlines(
            seg_peak_time_from_corr,
            ymin=np.min(seg_corrs),
            ymax=np.max(seg_corrs),
            color="orange",
            linestyle="--",
            alpha=0.75,
        )
        ax[2].set_title("Peaks initially detected from correlation")
        # Plot peak time from corr
        ax[3].plot(seg_corr_times, seg_corrs, alpha=0.75)
        ax[3].vlines(
            seg_peak_time_from_corr_rri_filtered,
            ymin=np.min(seg_corrs),
            ymax=np.max(seg_corrs),
            color="orange",
            linestyle="--",
            alpha=0.75,
        )

        ax[3].set_title("Filtered peaks")
        ax[4].plot(seg_corr_times, seg_corrs, alpha=0.75)
        ax[4].vlines(
            seg_final_peak_time,
            ymin=np.min(seg_corrs),
            ymax=np.max(seg_corrs),
            color="orange",
            linestyle="--",
            alpha=0.75,
        )
        ax[4].set_title("Final peaks")
        ax[4].set_xlabel("Time (seconds)")
        plt.tight_layout()
        plt.savefig(export_dir_root / "ecg_and_corr_with_peaks.pdf")


if __name__ == "__main__":
    main()
