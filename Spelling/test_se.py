import os
import time

import numpy as np
import spelling_errors as se
import pandas as pd
import ud_utils as udt
import pickle
import sys
import logging
import yaml

with open(sys.argv[1]) as config_file:
    config = yaml.load(config_file, Loader=yaml.SafeLoader)

logging.basicConfig(filename=config['log_path'] + '_app.log', filemode='w',
                    format='%(name)s - %(levelname)s - %(message)s', level=logging.INFO)

logging.info("SE Test started")
ground_truth = config['ground_truth_available']

with open(config['se_pkl_path'], 'rb') as f:
    se_dict = pickle.load(f)
    logging.info("se_dict loaded")
with open(config['test_pkl_path'], 'rb') as f:
    test = pickle.load(f)
    logging.info("test_dict loaded")
    
file_type = config['file_type']
output_path = config['output_path']
tables_output_path = os.path.join(output_path, "se_tables_test_results")
if not os.path.exists(tables_output_path):
    os.makedirs(tables_output_path)

spelling_results = pd.DataFrame(columns=['error_type', 'path', 'col_name', 'row_idx', 'col_idx', 'LR', 'value',
                                         'ground_truth', 'error'])
t_init = time.time()
for path in test:
    spelling_results_table = pd.DataFrame(columns=['error_type', 'path', 'col_name', 'row_idx', 'col_idx', 'LR', 'value',
                                        'ground_truth', 'error'])
    try:
        if file_type == "parquet":
            test_df = pd.read_parquet(path)
        else:
            test_df = pd.read_csv(path)
        
        for test_column_name in test_df.columns:
            test_column = test_df[test_column_name]
            mpd_d, mpd_do, avg_len_diff_tokens, idx_p = se.perturbation(test_column)
            p_dt, p_dot = 0, 0
            number_of_rows_range = udt.get_range_count(test_column.count())
            range_mpd = udt.get_range_mpd(avg_len_diff_tokens)

            if not np.isnan(mpd_d) and not np.isnan(mpd_do) and mpd_do != mpd_d:
                for col_id in se_dict.keys():
                    str_col = test_column.astype(str)
                    test_col_dtype = "alnumeric" if str_col.str.isalnum().all() else test_column.dtype,
                    if se_dict[col_id]["d_type"] != test_col_dtype:
                        continue
                    if se_dict[col_id]["number_of_rows_range"] != number_of_rows_range:
                        continue
                    if se_dict[col_id]["range_mpd"] != range_mpd:
                        continue

                    train_mpd_d, train_mpd_do = se_dict[col_id]["mpd"], se_dict[col_id]["mpd_p"]
                    if train_mpd_d <= mpd_do:
                        p_dt = p_dt + 1
                    if train_mpd_d <= mpd_d and train_mpd_do >= mpd_do:
                        p_dot = p_dot + 1
                lr = p_dot / p_dt if p_dt else -np.inf
                if ground_truth:
                    ground_truth_path = config['ground_truth_path']
                    clean_df = pd.read_csv(os.path.join(ground_truth_path, os.path.basename(path)))
                    correct_value = clean_df[test_column_name].loc[idx_p]
                else:
                    correct_value = "----Ground Truth is Not Available----"
                if idx_p and lr != -np.inf:
                    error = str(test_column.loc[idx_p]) != str(correct_value)
                    row = ["spelling", path, test_column_name, idx_p, list(test_df.columns).index(test_column_name), lr,
                        test_column.loc[idx_p], correct_value, error]
                                
                    spelling_results.loc[len(spelling_results)] = row
                    spelling_results_table.loc[len(spelling_results_table)] = row
        
        with open(os.path.join(tables_output_path, (os.path.basename(path).removesuffix(f'.{file_type}') + ".pickle")), 'wb') as f:
            pickle.dump(spelling_results_table, f)
    except Exception as e:
        logging.info(f"Error in {path}: {e}")
        continue

spelling_results.to_csv(os.path.join(config['output_path'], "se_test_results.csv"))
t_f = time.time() - t_init
logging.info(f"SE Test finished in {t_f} seconds")