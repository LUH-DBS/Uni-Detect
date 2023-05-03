import os
import time

import numpy as np
import numeric_outliers as no
import pandas as pd
import ud_utils as udt
import pickle
import logging
import sys
import yaml

# Config file
with open(sys.argv[1]) as config_file:
    config = yaml.load(config_file, Loader=yaml.SafeLoader)

logging.basicConfig(filename=config['log_path'] + '_app.log', filemode='w',
                    format='%(name)s - %(levelname)s - %(message)s', level=logging.INFO)

t0 = time.time()
logging.info("Start test_no.py")
ground_truth = config['ground_truth_available']
with open(config['test_pkl_path'], 'rb') as f:
    test = pickle.load(f)
    logging.info("test_dict loaded")
with open(config['no_dict'], 'rb') as f:
    no_dict = pickle.load(f)
    logging.info("no_dict loaded")

no_results = pd.DataFrame(
    columns=['error_type', 'path', 'col_name', 'row_idx', 'col_idx', 'LR', 'value', 'correct_value', 'error'])
file_type = config['file_type']
output_path = config['output_path']
tables_output_path = os.path.join(output_path, "no_tables_test_results")
if not os.path.exists(tables_output_path):
    os.makedirs(tables_output_path)

for path in test:
    try:
        # Load test table and get numeric columns
        test_df = pd.read_csv(path).select_dtypes(include=[np.number])
        for test_column_name in test_df.columns:
            p_dt, p_t_dt, p_dot, p_t_dot, max_idx, max_idx_t = 0, 0, 0, 0, -1, -1
            try:
                test_column = test_df[test_column_name]
                max_mad_test_d, max_mad_test_do, max_idx = no.perturbation(test_column)
                max_idx_t = -1
                number_of_rows_range = udt.get_range_count(test_column.count())
            except Exception as e:
                logging.info(f"Error in {path} {test_column_name}")
                continue

            # Compare the test column with the train columns
            if max_mad_test_d != max_mad_test_do:
                for col_id in no_dict.keys():
                    if no_dict[col_id]['max_mad'] is not None:
                        # Check the data type
                        if no_dict[col_id]["d_type"] != test_column.dtype:
                            continue
                        # Check the number of row range
                        if no_dict[col_id]["number_of_rows_range"] != number_of_rows_range:
                            continue

                        max_mad_train_d, max_mad_train_do = no_dict[col_id]["max_mad"], no_dict[col_id]["max_mad_p"]
                        tr_p = no.perturbation(no_dict[col_id]["col_transformed"])

                        # Checking whether the log transform better fits the data
                        if tr_p != (None, None, None):
                            max_mad_train_transformed_d, max_mad_train_transformed_do, max_idx_t = \
                                tr_p[0], tr_p[1], tr_p[2]
                        else:
                            max_mad_train_transformed_d, max_mad_train_transformed_do = np.inf, np.inf

                        
                        if max_mad_train_d >= max_mad_test_do:
                            p_dt = p_dt + 1
                        # before perturbation the max_mad is greater than or equal to max_mad_c(theta_1) and
                        # after perturbation the max_mad is less than or equal to max_mad_c(theta_2)
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
            if ground_truth:
                ground_truth_path = config['ground_truth_path']
                clean_df = pd.read_csv(os.path.join(ground_truth_path, os.path.basename(path)))
                correct_value = clean_df[test_column_name].loc[max_idx]
            else:
                correct_value = "----Ground Truth is Not Available----"

            if max_idx != -1:
                row = ["no", path, test_column_name, max_idx, list(test_df.columns).index(test_column_name),
                        lr,
                        test_column.loc[max_idx], correct_value, correct_value != test_column.loc[max_idx]]
            else:
                row = ["no", path, test_column_name, max_idx, list(test_df.columns).index(test_column_name),
                       None, None, None, None]
        else:
            row = ["no", path, test_column_name, None, list(test_df.columns).index(test_column_name),
                       None, None, None, None]
                       
            no_results.loc[len(no_results)] = row

        with open(os.path.join(tables_output_path, (os.path.basename(path).removesuffix(f'.{file_type}') + ".pickle")), 'wb') as f:
            pickle.dump(no_results, f)
    except Exception as e:
        logging.info(f"Error in {path}: {e}")
no_results.to_csv(os.path.join(config['output_path'], "no_test_results.csv"))
t1 = time.time()
logging.info(f"no test time: {t1-t0}") 
