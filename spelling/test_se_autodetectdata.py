import logging
import os
import pickle
import sys
import time

import numpy as np
import pandas as pd
import yaml

import utils.ud_utils as udt
from spelling import spelling_errors as se

with open(sys.argv[1]) as config_file:
    config = yaml.load(config_file, Loader=yaml.SafeLoader)

os.makedirs(config["log_path"], exist_ok=True)
logging.basicConfig(
    filename=config["log_path"] + "/app.log",
    filemode="w",
    format="%(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)

with open(config["se_pkl_path"], "rb") as f:
    se_dict = pickle.load(f)
    logging.info("se_dict loaded")
with open(config["test_pkl_path"], "rb") as f:
    test = pickle.load(f)
    logging.info("test_dict loaded")
spelling_results = pd.DataFrame(
    columns=[
        "error_type",
        "path",
        "col_name",
        "row_idx",
        "col_idx",
        "LR",
        "value",
        "Label",
        "error",
        "time",
    ]
)
for path in test:
    spelling_results_table = pd.DataFrame(
        columns=[
            "error_type",
            "path",
            "col_name",
            "row_idx",
            "col_idx",
            "LR",
            "value",
            "Label",
            "error",
            "time",
        ]
    )

    t0 = time.time()
    df = pd.read_csv(path)
    to_be_dropped = df.select_dtypes([np.number])
    test_df = df.drop(to_be_dropped, axis=1)
    for test_column_name in test_df.columns:
        if test_column_name != "Label":
            try:
                test_column = test_df[test_column_name]
                mpd_d, mpd_do, avg_len_diff_tokens, idx_p = se.perturbation(test_column)
                p_dt, p_dot = 0, 0
                number_of_rows_range = udt.get_range_count(test_column.count())
                range_mpd = udt.get_range_mpd(avg_len_diff_tokens)

            except Exception as e:
                print(e)
                continue

            if mpd_do != mpd_d:
                for col_id in se_dict.keys():
                    if se_dict[col_id]["d_type"] != test_column.dtype:
                        continue
                    if se_dict[col_id]["number_of_rows_range"] != number_of_rows_range:
                        continue
                    if se_dict[col_id]["range_mpd"] != range_mpd:
                        continue

                    train_mpd_d, train_mpd_do = (
                        se_dict[col_id]["mpd"],
                        se_dict[col_id]["mpd_p"],
                    )
                    if train_mpd_d <= mpd_do:
                        p_dt = p_dt + 1
                    if train_mpd_d <= mpd_d and train_mpd_do >= mpd_do:
                        p_dot = p_dot + 1
                lr = p_dot / p_dt if p_dt else -np.inf
                t1 = time.time()
                if idx_p and lr != -np.inf:
                    label = test_df["label"].loc[idx_p]
                    if label == 2:
                        error = 1
                    elif label == 1:
                        error = 0
                    else:
                        error = -1
                    row = [
                        "spelling",
                        path,
                        test_column_name,
                        idx_p,
                        list(test_df.columns).index(test_column_name),
                        lr,
                        test_column.loc[idx_p],
                        label,
                        error,
                        t1 - t0,
                    ]
                    spelling_results.loc[len(spelling_results)] = row

spelling_results.to_csv(config["output_path"])
