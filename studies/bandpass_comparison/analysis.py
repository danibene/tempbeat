#!/usr/bin/env python

# import os
from pathlib import Path

import matplotlib.pyplot as plt

# import numpy as np
import pandas as pd
import scipy
import seaborn as sns
import soundfile as sf

from tempbeat.evaluation.compare_bpm import get_bpm_mae_from_peak_time
from tempbeat.extraction.heartbeat_extraction import hb_extract
from tempbeat.extraction.interval_conversion import peak_time_to_rri
from tempbeat.utils.interpolation import interpolate_to_same_x
from tempbeat.utils.matlab_utils import quit_matlab, set_matlab
from tempbeat.utils.timestamp import sampling_rate_to_sig_time

# from neurokit2 import data, ecg_process, signal_distort


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
    # This is how you can use Neurokit2 to process ECG

    set_matlab()
    root_path = Path("Z:/Shared/Documents/RD/RD2/_AudioRD/datasets/Biosignals")
    datasets = ["P5M5_1", "P5M5_2", "P5M5_3"]

    export_dir_root = Path("./output_2023-12-06")

    for minutes in [5, 6, 4, 3, 2, 1, 0.5, 0.25]:
        export_dir = export_dir_root / str(minutes * 60)

        if not export_dir.exists():
            export_dir.mkdir(parents=True)

        fig_export_dir = export_dir / "figs"

        if not fig_export_dir.exists():
            fig_export_dir.mkdir(parents=True)

        rows = []
        row_peaks_list = []
        for dataset in datasets:
            dataset_path = root_path / dataset / "8k"

            participants = [
                folder.name
                for folder in dataset_path.iterdir()
                if (folder.is_dir() & (folder.name[0] == "P"))
            ]

            for p in participants:
                for side in ["L", "R"]:
                    try:
                        iem_path = (
                            dataset_path / p / "_unsegmented" / ("IEM_" + side + ".wav")
                        )
                        ecg_path = dataset_path / p / "_unsegmented" / "ECG_audio.wav"
                        labels_path = (
                            dataset_path / p / "_unsegmented" / "mrkrConditions.csv"
                        )

                        labels = pd.read_csv(labels_path)

                        labels_start = labels[labels["y"] == "00-fitTestNoise1-start"]
                        start_time = labels_start.iloc[[-1]].t.values[0]
                        # labels_stop = labels[labels['y'] == '03-noiseIels-stop']
                        # stop_time = labels_stop.iloc[[-1]].t.values[0]
                        stop_time = start_time + (minutes * 60)

                        ecg_audio, sr = read_audio_section(
                            ecg_path, start_time, stop_time
                        )
                        clean_peak_time = [
                            i / sr for i, v in enumerate(ecg_audio) if v > 0.5
                        ]
                        iem_audio, sr = read_audio_section(
                            iem_path, start_time, stop_time
                        )
                        audio_sig_time = sampling_rate_to_sig_time(
                            iem_audio, sampling_rate=sr
                        )

                        new_sampling_rate = 1000
                        div = sr / new_sampling_rate
                        (
                            resampled_clean_sig,
                            resampled_clean_sig_time,
                        ) = scipy.signal.resample(
                            iem_audio, num=int(len(iem_audio) / div), t=audio_sig_time
                        )

                        new_sampling_rate = 100
                        div = 1000 / new_sampling_rate
                        (
                            resampled_clean_sig,
                            resampled_clean_sig_time,
                        ) = scipy.signal.resample(
                            resampled_clean_sig,
                            num=int(len(resampled_clean_sig) / div),
                            t=resampled_clean_sig_time,
                        )

                        MAE_list = []
                        peaks_list = []
                        hb_extract_methods = ["no_temp", "temp", "matlab"]
                        colors = ["red", "blue", "orange"]
                        for hb_extract_method in hb_extract_methods:
                            audio_peak_time = hb_extract(
                                resampled_clean_sig,
                                sig_time=resampled_clean_sig_time,
                                sampling_rate=new_sampling_rate,
                                method=hb_extract_method,
                            )
                            mae_clean_distorted = get_bpm_mae_from_peak_time(
                                peak_time_a=clean_peak_time, peak_time_b=audio_peak_time, unit="rri"
                            )

                            peaks_list.append(audio_peak_time)

                            # print(hb_extract_method)
                            # print(mae_clean_distorted)
                            MAE_list.append(mae_clean_distorted)

                        interpolation_rate = 2
                        min_bpm = 40
                        max_bpm = 200
                        min_rri = 60000 / max_bpm
                        max_rri = 60000 / min_bpm
                        rri_truth, rri_time_truth = peak_time_to_rri(
                            clean_peak_time, min_rri=min_rri, max_rri=max_rri
                        )

                        for i in range(len(hb_extract_methods)):
                            rri_a, rri_time_a = peak_time_to_rri(
                                peaks_list[i], min_rri=min_rri, max_rri=max_rri
                            )

                            interp_x, interp_a, interp_truth = interpolate_to_same_x(
                                a_x=rri_time_a,
                                a_y=rri_a,
                                b_x=rri_time_truth,
                                b_y=rri_truth,
                                interpolation_rate=interpolation_rate,
                            )
                            sns.lineplot(
                                x=interp_x,
                                y=interp_a,
                                color=colors[i],
                                label=hb_extract_methods[i],
                            )

                        sns.lineplot(
                            x=interp_x,
                            y=interp_truth,
                            color="green",
                            label="Ground Truth",
                        )

                        plt.title(f"{dataset} {p} {side}")
                        plt.xlabel("Time")
                        plt.ylabel("RRI")
                        plt.legend()

                        fn = f"{dataset}-{p}-{side}.png"
                        plt.savefig(fig_export_dir / fn, bbox_inches="tight")

                        plt.show()

                        row = {
                            "dataset": dataset,
                            "participant": p,
                            "side": side,
                            "truth nbPeaks": len(clean_peak_time),
                            "no_temp MAE": MAE_list[0],
                            "no_temp nbPeaks": len(peaks_list[0]),
                            "temp MAE": MAE_list[1],
                            "temp nbPeaks": len(peaks_list[1]),
                            "matlab MAE": MAE_list[2],
                            "matlab nbPeaks": len(peaks_list[2]),
                        }

                        row_peaks = {
                            "dataset": dataset,
                            "participant": p,
                            "side": side,
                            "no_temp peaks": peaks_list[0],
                            "temp peaks": peaks_list[1],
                            "matlab peaks": peaks_list[2],
                        }

                        print(row)
                        rows.append(row)
                        row_peaks_list.append(row_peaks)

                    except Exception as e:
                        print(f"{dataset} - {p} - {side} \nAn error occurred: {e}")

        df_final = pd.DataFrame(rows)

        df_final.to_csv(export_dir / f"results_{minutes*60}.csv", index=False)

        df_peaks = pd.DataFrame(row_peaks_list)
        df_peaks.to_json(export_dir / f"peak_output_{minutes*60}.json", orient="index")

    quit_matlab()


if __name__ == "__main__":
    main()
