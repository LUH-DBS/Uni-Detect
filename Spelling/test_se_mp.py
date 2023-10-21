from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Pool
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

def mapper(args):
    col_id, se_dict_entry, shared_args = args
    test_column, test_column_name, test_column_idx, mpd_d, mpd_do, number_of_rows_range, range_mpd = shared_args

    str_col = test_column.astype(str)
    test_col_dtype = "alnumeric" if str_col.str.isalnum().all() else test_column.dtype
    
    p_dt, p_dot = 0, 0
    if se_dict_entry["d_type"] != test_col_dtype:
        return (p_dt, p_dot)
    if se_dict_entry["number_of_rows_range"] != number_of_rows_range:
        return (p_dt, p_dot)
    if se_dict_entry["range_mpd"] != range_mpd:
        return (p_dt, p_dot)
    
    train_mpd_d, train_mpd_do = se_dict_entry["mpd"], se_dict_entry["mpd_p"]
    if train_mpd_d <= mpd_do:
        p_dt += 1
    if train_mpd_d <= mpd_d and train_mpd_do >= mpd_do:
        p_dot += 1
    return (p_dt, p_dot)

def reducer(results):
    total_p_dt, total_p_dot = 0, 0
    for result in results:
        total_p_dt += result[0]
        total_p_dot += result[1]
    lr = total_p_dot / total_p_dt if total_p_dt else -np.inf
    return lr

def process_column_mapreduce(test_column, se_dict, ground_truth, config, path, test_df, test_column_name, test_column_idx):
    rows = []
    try:
        # The previous part of your code remains unchanged:
        mpd_d, mpd_do, avg_len_diff_tokens, idx_p = se.perturbation(test_column)
        number_of_rows_range = udt.get_range_count(test_column.count())
        range_mpd = udt.get_range_mpd(avg_len_diff_tokens)
        
        # Prepare shared arguments for mappers
        shared_args = (test_column, test_column_name, test_column_idx, mpd_d, mpd_do, number_of_rows_range, range_mpd)
        
        # Start the map phase
        with Pool() as pool:
            map_results = pool.map(mapper, [(col_id, se_dict[col_id], shared_args) for col_id in se_dict.keys()])
            
        # Apply the reduce phase
        lr = reducer(map_results)
        
        # The rest of your code to gather results remains unchanged:
        rows = []
        if ground_truth:
            ground_truth_path = config['ground_truth_path']
            clean_df = pd.read_csv(os.path.join(ground_truth_path, os.path.basename(path)))
            test_column_name_clean = clean_df.columns[test_column_idx]
            correct_value = clean_df[test_column_name_clean].values.astype(str)[idx_p]
            dirty_value = test_df[test_df.columns[test_column_idx]].values.astype(str)[idx_p]
        else:
            correct_value = "----Ground Truth is Not Available----"
        if idx_p and lr != -np.inf:
            error = str(correct_value) != str(dirty_value)
            row = ["spelling", path, test_column_name, idx_p, list(test_df.columns).index(test_column_name), lr,
                test_column.loc[idx_p], correct_value, error]
            rows.append(row)
    except Exception as e:
        logging.info(f"Error in table {path} column {test_column_name} - Error: {e}")
    return rows

def process_column(test_column, se_dict, ground_truth, config, path, test_df, test_column_name, test_column_idx):
    mpd_d, mpd_do, avg_len_diff_tokens, idx_p = se.perturbation(test_column)
    p_dt, p_dot = 0, 0
    number_of_rows_range = udt.get_range_count(test_column.count())
    range_mpd = udt.get_range_mpd(avg_len_diff_tokens)
    rows = []

    if not np.isnan(mpd_d) and not np.isnan(mpd_do) and mpd_do != mpd_d:
        for col_id in se_dict.keys():
            str_col = test_column.astype(str)
            if str_col.str.isnumeric().all():
                test_col_dtype = "alnumeric"
            else:
                test_col_dtype = test_column.dtype
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
            test_column_name_clean = clean_df.columns[test_column_idx]
            correct_value = clean_df[test_column_name_clean].values.astype(str)[idx_p]
            dirty_value = test_df[test_df.columns[test_column_idx]].values.astype(str)[idx_p]
        else:
            correct_value = "----Ground Truth is Not Available----"
        if idx_p and lr != -np.inf:
            error = str(correct_value) != str(dirty_value)
            row = ["spelling", path, test_column_name, idx_p, list(test_df.columns).index(test_column_name), lr,
                test_column.loc[idx_p], correct_value, error]
                        
            rows.append(row)
    return rows
        
        
def main():
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
        logging.info(f"Processing {path}")
        spelling_results_table = pd.DataFrame(columns=['error_type', 'path', 'col_name', 'row_idx', 'col_idx', 'LR', 'value',
                                            'ground_truth', 'error'])
        results = []
        try:
            if file_type == "parquet":
                test_df = pd.read_parquet(path)
            else:
                test_df = pd.read_csv(path)
            
            for test_column_idx, test_column_name in enumerate(test_df.columns):
                    logging.info(f"Processing {test_column_name}")
                    test_column = test_df[test_column_name]
                    results.append(process_column_mapreduce(test_column, se_dict, ground_truth, config, path, test_df, test_column_name, test_column_idx))
            for rows in results:
                for row in rows:
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

    return

if __name__ == "__main__":
    main()