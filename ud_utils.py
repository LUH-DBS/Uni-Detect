from multiprocessing import Pool, cpu_count
import pandas as pd
import numpy as np
from Numeric_Outliers import numeric_outliers as no
from Spelling import spelling_errors as se
from uv import uniqueness as uv
import pickle
from itertools import combinations
import logging

from fd_violations import fd


def get_range_count(num_rows):
    if num_rows <= 20:
        return 0
    if num_rows <= 50:
        return 1
    if num_rows <= 100:
        return 2
    if num_rows <= 500:
        return 3
    if num_rows <= 1000:
        return 4
    return 5


def get_range_mpd(mpd_diff):
    if mpd_diff <= 5:
        return 0
    if mpd_diff <= 10:
        return 1
    if mpd_diff <= 15:
        return 2
    if mpd_diff <= 20:
        return 3
    return 4


def get_range_avg_pre(avg_tokens):
    if avg_tokens <= 50:
        return 0
    if avg_tokens <= 100:
        return 1
    if avg_tokens <= 1000:
        return 2
    if avg_tokens <= 10000:
        return 3
    if avg_tokens <= 100000:
        return 4
    return 5


def get_tokens_dict(train_path):
    tokens_dict = uv.offline_learning_corpus(train_path)
    with open('pkl/tokens_dict.pkl', 'wb') as f:
        pickle.dump(tokens_dict, f)
    return tokens_dict

# Define the function to be applied to each path
def process_path(path, file_type):
    try:
        if file_type == "parquet":
            train_df = pd.read_parquet(path + "/clean.parquet")
        else:
            train_df = pd.read_csv(path)
        
        # numeric outliers
        train_df_no = train_df.select_dtypes(include=[np.number])
        path_no_dict = {}
        for col_name in train_df_no.columns:
            col_id = path + "_" + col_name
            col = train_df_no[col_name]
            # Ignore empty cols
            if not col.empty:
                # Get column measures
                path_no_dict[col_id] = no.get_col_measures(col)
        logging.info(f"df shape: {train_df.shape}")
        return path_no_dict
    except Exception as e:
        logging.error(f"Error processing path {path}: {e}")
        return {}
        
def no_offline_learning(train, file_type, output_path):
    no_dict = {}

    # Use multiprocessing to apply the function to each path in parallel
    with Pool(processes=cpu_count()) as p:
        path_no_dicts = p.starmap(process_path, ((path, file_type) for path in train))
        
    # Merge the results into a single dictionary
    for path_no_dict in path_no_dicts:
        no_dict.update(path_no_dict)
        
    # Save the dictionary to disk
    with open(output_path, 'wb') as f:
        pickle.dump(no_dict, f)
    return no_dict


def se_offline_learning(train, file_type, output_path):
    # Number of processed columns
    count = 0
    se_dict = {}
    for path in train:
        try:
            if file_type == "parquet":
                train_df = pd.read_parquet(path + "/clean.parquet")
            else:
                train_df = pd.read_csv(path)
            # Select non-numeric cols
            to_be_dropped = train_df.select_dtypes([np.number])
            train_df_se = train_df.drop(to_be_dropped, axis=1)
            for col_name in train_df_se.columns:
                col_id = path + "_" + col_name
                col = train_df_se[col_name]
                # Ignore empty cols
                if not col.empty:
                    # Get column measures
                    se_dict[col_id] = se.get_col_measures(col)
                count += 1
                if count % 100 == 0:
                    logging.info(f"se_count: {count}")
            logging.info(f"df shape: {train_df.shape}")
        except Exception as e:
            logging.error(e, path)
    # Save results
    with open(output_path, 'wb') as f:
        pickle.dump(se_dict, f)
    return se_dict


def uv_offline_learning(train, file_type, output_path):
    # Number of processed columns
    count = 0
    uv_dict = {}
    tokens_dict = get_tokens_dict(train)

    for path in train:
        try:
            if file_type == "parquet":
                train_df = pd.read_parquet(path + "/clean.parquet")
            else:
                train_df = pd.read_csv(path)
            for col_name in train_df.columns:
                col = train_df[col_name]
                if not col.empty:
                    col_id = path + "_" + col_name
                    uv_dict[col_id] = uv.get_col_measures(train_df[col_name], list(train_df.columns).index(col_name),
                                                          tokens_dict)
            count += 1
            if count % 100 == 0:
                logging.info(f"uv_count: {count}")
            logging.info(f"df shape: {train_df.shape}")
        except Exception as e:
            logging.error(e, path)
    with open(output_path, 'wb') as f:
        pickle.dump(uv_dict, f)


def fd_offline_learning(train, file_type, output_path):
    # Number of processed columns
    count = 0
    fd_dict = {}
    tokens_dict = get_tokens_dict(train)

    for path in train:
        try:
            if file_type == "parquet":
                train_df = pd.read_parquet(path + "/clean.parquet")
            else:
                train_df = pd.read_csv(path)
        # functional dependencies
            for pair in combinations(train_df.columns, 2):
                if pair[0] != pair[1]:
                    cols_id = path + "_" + pair[0] + pair[1]
                    fd_dict[cols_id] = fd.get_col_measures(train_df[pair[0]], train_df[pair[1]],
                                                           list(train_df.columns).index(pair[0]),
                                                           list(train_df.columns).index(pair[1]), tokens_dict)
            count += 1
            if count % 100 == 0:
                logging.info(f"fd_count: {count}")
            logging.info(f"df shape: {train_df.shape}")
        except Exception as e:
            logging.error(e, path)
    with open(output_path, 'wb') as f:
        pickle.dump(fd_dict, f)
    return tokens_dict, fd_dict
