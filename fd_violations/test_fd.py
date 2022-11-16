import time
import numpy as np
import fd
import pandas as pd
from uv import uniqueness as uv
import ud_utils as udt
import pickle
from itertools import combinations
import logging
import sys
import yaml

with open(sys.argv[1]) as config_file:
    config = yaml.load(config_file, Loader=yaml.SafeLoader)

logging.basicConfig(filename=config['log_path'] + '_app.log', filemode='w',
                    format='%(name)s - %(levelname)s - %(message)s', level=logging.INFO)

with open(config['test_pkl_path'], 'rb') as f:
    test = pickle.load(f)
    test = [path.replace("/home/fatemeh/GitTables/Sandbox/", "/Users/fatemehahmadi/Documents/Sandbox/") for path in test]
    logging.info("test_dict loaded")

with open(config['fd_pkl_path'], 'rb') as f:
    fd_dict = pickle.load(f)
    logging.info("fd_dict loaded")

with open(config['tokens_pkl_path'], 'rb') as f:
    tokens_dict = pickle.load(f)
    print("tokens_dict loaded")

fd_results = pd.DataFrame(
    columns=['error_type', 'path', 'col_1_name', 'col_2_name', 'row_idx', 'col_1_idx', 'col_2_idx', 'LR',
             'value_1', 'value_2', 'correct_value_1', 'correct_value_2', 'error'])
test_column_1, test_column_2, left_ness_1, left_ness_2, fd_d, fd_do, idx_d = np.nan, np.nan, np.nan, np.nan,\
                                                                             np.nan, np.nan, np.nan
for path in test:
    t0 = time.time()
    try:
        test_df = pd.read_csv(path + "/dirty.csv")
        for pair in combinations(test_df.columns, 2):
            test_column_1, test_column_2 = test_df[pair[0]], test_df[pair[1]]
            fd_d, fd_do, idx_d = fd.perturbation(test_column_1, test_column_2)
            p_dt, p_dot = 0, 0
            if idx_d != -1 and fd_do != fd_d:
                left_ness_1 = list(test_df.columns).index(pair[0])
                left_ness_2 = list(test_df.columns).index(pair[1])
                number_of_rows_range = udt.get_range_count(test_column_1.count())
                avg_col_pre_1 = uv.get_prev_range(tokens_dict, test_column_1)
                avg_col_pre_2 = uv.get_prev_range(tokens_dict, test_column_2)

                for cols_id in fd_dict.keys():
                    if fd_dict[cols_id]["d_type_1"] != test_column_1.dtype or\
                            fd_dict[cols_id]["d_type_2"] != test_column_2.dtype:
                        continue
                    if fd_dict[cols_id]["number_of_rows_range"] != number_of_rows_range:
                        continue
                    if fd_dict[cols_id]["left_ness_1"] != left_ness_1 or fd_dict[cols_id]["left_ness_2"] != left_ness_2:
                        continue
                    if fd_dict[cols_id]["avg_col_pre_1"] != avg_col_pre_1 or\
                            fd_dict[cols_id]["avg_col_pre_2"] != avg_col_pre_2 :
                        continue

                    train_fd_d, train_fd_do = fd_dict[cols_id]["fd"], fd_dict[cols_id]["fd_p"]
                    if train_fd_d != train_fd_do:
                        if train_fd_d <= fd_do:
                            p_dt = p_dt + 1
                        if train_fd_d <= fd_d and train_fd_do >= fd_do:
                            p_dot = p_dot + 1

            lr = p_dot / p_dt if p_dt and p_dot else -np.inf
            if lr != -np.inf:
                clean_df = pd.read_excel(path + "/clean.parquet")
                if idx_d != -1:
                    correct_value_1 = clean_df[pair[0]].loc[idx_d]
                    correct_value_2 = clean_df[pair[1]].loc[idx_d]

                    row = ["fd", path, pair[0], pair[1], idx_d, list(test_df.columns).index(pair[0]),
                           list(test_df.columns).index(pair[1]),
                           lr, test_column_1.loc[idx_d], test_column_2.loc[idx_d],
                           correct_value_1, correct_value_2,
                           correct_value_1 != test_column_1.loc[idx_d] or correct_value_2 != test_column_2.loc[idx_d]]
                    fd_results.loc[len(fd_results)] = row
    except Exception as e:
        print(e)

fd_results.to_csv(config['output_path'])
