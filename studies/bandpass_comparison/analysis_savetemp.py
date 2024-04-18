#!/usr/bin/env python

from pathlib import Path

import pandas as pd
import scipy
import soundfile as sf
import seaborn as sns
import matplotlib.pyplot as plt

from tempbeat.evaluation.compare_bpm import get_bpm_mae_from_peak_time
from tempbeat.extraction.heartbeat_extraction import hb_extract
from tempbeat.extraction.interval_conversion import peak_time_to_rri
from tempbeat.utils.matlab_utils import quit_matlab, set_matlab
from tempbeat.utils.timestamp import sampling_rate_to_sig_time
from tempbeat.utils.interpolation import interpolate_to_same_x


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

    # set_matlab()
    print("Matlab Started")
    root_path = Path("Z:/Shared/Documents/RD/RD2/_AudioRD/datasets/Biosignals")
    datasets = ["P5M5_1", "P5M5_2", "P5M5_3"]
    # datasets = ["P5M5_2"]

    export_dir_root = Path("./output_2024-04-09")
    mins = [5, 3,  1, 0.5, 0.25]
    mins.reverse()
    minutes = 5
    export_dir = export_dir_root / str(minutes * 60)

    if not export_dir.exists():
        export_dir.mkdir(parents=True)

    fig_export_dir = export_dir / "figs"

    if not fig_export_dir.exists():
        fig_export_dir.mkdir(parents=True)

    rows_bpm_p = []
    rows_bpm = []
    rows_rri_p = []
    rows_rri = []

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
                    ecg_path = dataset_path / p / "_unsegmented" / "ECG_audio2.wav"

                    if not ecg_path.is_file():
                        ecg_path = (
                            dataset_path / p / "_unsegmented" / "ECG_audio.wav"
                        )
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

                    fig = plt.figure()        
                    hb_extract_method = "temp"
                    for minutes in mins:
                        if hb_extract_method == "temp":
                            hb_extract_kwargs = {"output_format": "full"}
                        else:
                            hb_extract_kwargs = {}
                        output = hb_extract(
                            resampled_clean_sig[0:int(minutes*60*new_sampling_rate)],
                            sig_time=resampled_clean_sig_time[0:int(minutes*60*new_sampling_rate)],
                            sampling_rate=new_sampling_rate,
                            method=hb_extract_method,
                            hb_extract_algo_kwargs=hb_extract_kwargs,
                        )
                        if isinstance(output, tuple):
                            audio_peak_time = output[0]
                            med_template = output[1]["med_template"]
                        else:
                            audio_peak_time = output

                    
                    
                        plt.plot(med_template)
                    plt.legend((np.array(mins)*60).astype(int))
                        
                    plt.xlabel("Samples")
                    fig.tight_layout()
                    fn = f"{dataset}-{p}-{side}--Template.pdf"
                    plt.savefig(fig_export_dir / fn, bbox_inches="tight")
                    plt.show()
                    


                except Exception as e:
                    print(f"{dataset} - {p} - {side} \nAn error occurred: {e}")


    # quit_matlab()


if __name__ == "__main__":
    main()
