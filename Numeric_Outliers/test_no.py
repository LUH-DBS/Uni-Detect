import time

import numpy as np
from Numeric_Outliers import numeric_outliers as no
import pandas as pd
import ud_utils as udt
import pickle
import logging
import sys
import yaml

with open(sys.argv[1]) as config_file:
    config = yaml.load(config_file, Loader=yaml.SafeLoader)

logging.basicConfig(filename=config['log_path'] + '_app.log', filemode='w',
                    format='%(name)s - %(levelname)s - %(message)s', level=logging.INFO)

t0 = time.time()
with open(config['test_pkl_path'], 'rb') as f:
    test = pickle.load(f)
    test = [path.replace("/home/fatemeh/GitTables/Sandbox/", "/Users/fatemehahmadi/Documents/Sandbox/") for path in test]
    logging.info("test_dict loaded")
with open(config['no_dict'], 'rb') as f:
    no_dict = pickle.load(f)
    logging.info("no_dict loaded")

no_results = pd.DataFrame(
    columns=['error_type', 'path', 'col_name', 'row_idx', 'col_idx', 'LR', 'value', 'correct_value', 'error'])

for path in test:
    try:
        test_df = pd.read_csv(path + "/dirty.csv").select_dtypes(include=[np.float, np.int])
        for test_column_name in test_df.columns:
            p_dt, p_t_dt, p_dot, p_t_dot, max_idx, max_idx_t = 0, 0, 0, 0, -1, -1
            try:
                test_column = test_df[test_column_name]
                max_mad_test_d, max_mad_test_do, max_idx = no.perturbation(test_column)
                max_idx_t = -1
                number_of_rows_range = udt.get_range_count(test_column.count())
            except Exception as e:
                print(e)
                continue

            if max_mad_test_d != max_mad_test_do:
                for col_id in no_dict.keys():
                    if no_dict[col_id]["d_type"] != test_column.dtype:
                        continue
                    if no_dict[col_id]["number_of_rows_range"] != number_of_rows_range:
                        continue

                    max_mad_train_d, max_mad_train_do = no_dict[col_id]["max_mad"], no_dict[col_id]["max_mad_p"]
                    tr_p = no.perturbation(no_dict[col_id]["col_transformed"])
                    if tr_p:
                        max_mad_train_transformed_d, max_mad_train_transformed_do, max_idx_t = \
                            tr_p[0], tr_p[1], tr_p[2]
                    else:
                        max_mad_train_transformed_d, max_mad_train_transformed_do = np.inf, np.inf

                    if max_mad_train_d >= max_mad_test_do:
                        p_dt = p_dt + 1
                    if max_mad_train_d >= max_mad_test_d and max_mad_train_do <= max_mad_test_do:
                        p_dot = p_dot + 1

                    if max_mad_train_transformed_d >= max_mad_test_do:
                        p_t_dt = p_t_dt + 1
                    if max_mad_train_transformed_d >= max_mad_test_d and max_mad_train_transformed_do <= max_mad_test_do:
                        p_t_dot = p_t_dot + 1

                lr = p_dot / p_dt if p_dt else -np.inf
                lr_t = p_t_dot / p_t_dt if p_t_dt else -np.inf

                if lr < lr_t:
                    lr = lr_t
                    max_idx = max_idx_t

                clean_df = pd.read_parquet(path + "/clean.parquet")
                if max_idx != -1:
                    correct_value = clean_df[test_column_name].loc[max_idx]
                    row = ["no", path, test_column_name, max_idx, list(test_df.columns).index(test_column_name),
                           lr,
                           test_column.loc[max_idx], correct_value, correct_value != test_column.loc[max_idx]]
                    no_results.loc[len(no_results)] = row

    except Exception as e:
        print(e)
no_results.to_csv(config['output_path'])
t1 = time.time()
print("no", t1-t0)
