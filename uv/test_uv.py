import numpy as np
import uniqueness as uv
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

with open(config['uv_pkl_path'], 'rb') as f:
    uv_dict = pickle.load(f)
    logging.info("uv_dict loaded")

with open(config['tokens_pkl_path'], 'rb') as f:
    tokens_dict = pickle.load(f)
    logging.info("tokens_dict loaded")

with open(config['test_pkl_path'], 'rb') as f:
    test = pickle.load(f)
    test = [path.replace("/home/fatemeh/GitTables/Sandbox/", "/Users/fatemehahmadi/Documents/Sandbox/") for path in test]
    logging.info("test_dict loaded")

uniqueness_results = pd.DataFrame(
    columns=['error_type', 'path', 'col_name', 'row_idx', 'col_idx', 'LR', 'value', 'correct_value', 'error'])

for path in test:
    try:
        test_df = pd.read_csv(path + "/dirty.csv")
        for test_column_name in test_df.columns:
            test_column = test_df[test_column_name]
            left_ness = list(test_df.columns).index(test_column_name)
            uniqueness_d, uniqueness_do, duplicate_idx = uv.perturbation(test_column)
            number_of_rows_range = udt.get_range_count(test_column.count())
            avg_col_pre = uv.get_prev_range(tokens_dict, test_column)

            p_dt, p_dot = 0, 0
            if duplicate_idx != -1 and uniqueness_d != uniqueness_do:
                for col_id in uv_dict.keys():
                    if uv_dict[col_id]["d_type"] != test_column.dtype:
                        continue
                    if uv_dict[col_id]["number_of_rows_range"] != number_of_rows_range:
                        continue
                    if uv_dict[col_id]["left_ness"] != left_ness:
                        continue
                    if uv_dict[col_id]["avg_col_pre"] != avg_col_pre:
                        continue

                    train_uniqueness_d, train_uniqueness_do = uv_dict[col_id]["ur"], \
                                                              uv_dict[col_id]["ur_p"]

                    if train_uniqueness_d <= uniqueness_do:
                        p_dt = p_dt + 1
                    if train_uniqueness_d <= uniqueness_d and train_uniqueness_do >= uniqueness_do:
                        p_dot = p_dot + 1

                lr = p_dot / p_dt if p_dt else -np.inf
                clean_df = pd.read_parquet(path + "/clean.parquet")
                correct_value = clean_df[test_column_name].loc[duplicate_idx]
                row = ["uniqueness", path, test_column_name, duplicate_idx,
                       list(test_df.columns).index(test_column_name), lr,
                       test_column.loc[duplicate_idx], correct_value, correct_value != test_column.loc[duplicate_idx]]
                uniqueness_results.loc[len(uniqueness_results)] = row
    except Exception as e:
        logging.info(e)

uniqueness_results.to_csv(config['output_path'])
