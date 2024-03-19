from pathlib import Path
from typing import List

import pandas as pd

export_dir_root = Path("./output_2024-03-14")

tests = [folder.name for folder in export_dir_root.iterdir() if folder.is_dir()]

remove_rows: List[dict] = [
    # {"dataset": "P5M5_1", "participant": "P17"},
    # {"dataset": "P5M5_1", "participant": "P18"},
    # {"dataset": "P5M5_1", "participant": "P24"},
    # {"dataset": "P5M5_1", "participant": "P25"},
    # {"dataset": "P5M5_1", "participant": "P26"},
    # {"dataset": "P5M5_2", "participant": "P02"},
    # {"dataset": "P5M5_2", "participant": "P03"},
    # {"dataset": "P5M5_2", "participant": "P06"},
    # {"dataset": "P5M5_2", "participant": "P08"},
    # {"dataset": "P5M5_2", "participant": "P11"},
    # {"dataset": "P5M5_2", "participant": "P15"},
    # {"dataset": "P5M5_3", "participant": "P01"},
    # {"dataset": "P5M5_3", "participant": "P03"},
    # {"dataset": "P5M5_2", "participant": "P05"},
    # {"dataset": "P5M5_2", "participant": "P06"},
]

remove_table = pd.DataFrame(remove_rows)

drop_bad = False

metrics = ["bpm_p", "bpm", "rri_p", "rri"]
columns_to_analyze = ["no_temp MAE", "temp MAE", "matlab MAE"]

for metric in metrics:
    rows = []
    for test in tests:
        df = pd.read_csv(export_dir_root / test / f"results_{metric}_{test}.csv")

        if drop_bad:
            merged_table = pd.merge(
                df,
                remove_table,
                on=["dataset", "participant"],
                how="left",
                indicator=True,
            )

            df = merged_table[merged_table["_merge"] == "left_only"].drop(
                columns="_merge"
            )

        mean_values = df[columns_to_analyze].mean().to_numpy()
        std_values = df[columns_to_analyze].std().to_numpy()

        row = {}

        row["length"] = int(float(test))

        for c in range(len(columns_to_analyze)):
            row[f"{columns_to_analyze[c]} mean"] = mean_values[c]
            row[f"{columns_to_analyze[c]} std"] = std_values[c]

        rows.append(row)

    df_final = pd.DataFrame(rows)
    df_final = df_final.sort_values(by="length")

    df_final.to_csv(export_dir_root / f"results_{metric}_summary.csv", index=False)
