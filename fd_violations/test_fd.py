import os
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

logging.info("FD Test started")
ground_truth = config['ground_truth_available']

with open(config['fd_pkl_path'], 'rb') as f:
    fd_dict = pickle.load(f)
    logging.info("fd_dict loaded")

with open(config['tokens_pkl_path'], 'rb') as f:
    tokens_dict = pickle.load(f)
    print("tokens_dict loaded")

with open(config['test_pkl_path'], 'rb') as f:
    test = pickle.load(f)
    logging.info("test_dict loaded")

file_type = config['file_type']
output_path = config['output_path']
tables_output_path = os.path.join(output_path, "fd_tables_test_results")
if not os.path.exists(tables_output_path):
    os.makedirs(tables_output_path)

fd_results = pd.DataFrame(
    columns=['error_type', 'path', 'col_1_name', 'col_2_name', 'row_idx', 'col_1_idx', 'col_2_idx', 'LR',
             'value_1', 'value_2', 'correct_value_1', 'correct_value_2', 'error'])
test_column_1, test_column_2, left_ness_1, left_ness_2, fd_d, fd_do, idx_d = np.nan, np.nan, np.nan, np.nan,\
                                                                             np.nan, np.nan, np.nan
t_init = time.time()
for path in test:
    fd_results_table = pd.DataFrame(
    columns=['error_type', 'path', 'col_1_name', 'col_2_name', 'row_idx', 'col_1_idx', 'col_2_idx', 'LR',
             'value_1', 'value_2', 'correct_value_1', 'correct_value_2', 'error'])
    try:
        if file_type == "parquet":
            test_df = pd.read_parquet(path)
        else:
            test_df = pd.read_csv(path)

        for pair in combinations(test_df.columns, 2):
            test_column_1, test_column_2 = test_df[pair[0]], test_df[pair[1]]
            fd_d, fd_do, idx_d = fd.perturbation(test_column_1, test_column_2)
            p_dt, p_dot = 0, 0
            if idx_d != -1 and fd_do != fd_d:
                left_ness_1 = list(test_df.columns).index(pair[0])
                left_ness_2 = list(test_df.columns).index(pair[1])
                number_of_rows_range = udt.get_range_count(test_column_1.count())
                avg_col_pre_1 = udt.get_prev_range(tokens_dict, test_column_1)
                avg_col_pre_2 = udt.get_prev_range(tokens_dict, test_column_2)

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
                    if idx_d != -1:
                        if ground_truth:
                            ground_truth_path = config['ground_truth_path']
                            clean_df = pd.read_csv(os.path.join(ground_truth_path, os.path.basename(path)))
                            clean_col_idx_0 = list(test_df.columns).index(pair[0])
                            clean_col_idx_1 = list(test_df.columns).index(pair[1])
                            correct_value_1 = clean_df[clean_df.columns[clean_col_idx_0]].values.astype(str)[idx_d]
                            correct_value_2 = clean_df[clean_df.columns[clean_col_idx_1]].values.astype(str)[idx_d]
                            dirty_value_1 = test_column_1.values.astype(str)[idx_d]
                            dirty_value_2 = test_column_2.values.astype(str)[idx_d]
                            
                        else:
                            correct_value_1 = "----Ground Truth is Not Available----"
                            correct_value_2 = "----Ground Truth is Not Available----"

                        row = ["fd", path, pair[0], pair[1], idx_d, list(test_df.columns).index(pair[0]),
                            list(test_df.columns).index(pair[1]),
                            lr, test_column_1.loc[idx_d], test_column_2.loc[idx_d],
                            correct_value_1, correct_value_2,
                            str(correct_value_1) != str(dirty_value_1) or str(correct_value_2) != str(dirty_value_2)]
                        fd_results.loc[len(fd_results)] = row
                        fd_results_table.loc[len(fd_results_table)] = row

            with open(os.path.join(tables_output_path, (os.path.basename(path).removesuffix(f'.{file_type}') + ".pickle")), 'wb') as f:
                pickle.dump(fd_results_table, f)
            logging.info(f"fd_test_results_table saved in {tables_output_path}")

    except Exception as e:
        logging.info(f"Error in {path}: {e}")

fd_results.to_csv(os.path.join(config['output_path'], "fd_test_results.csv"))
t_f = time.time() - t_init
logging.info(f"UV Test finished in {t_f} seconds")
