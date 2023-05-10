from concurrent.futures import ThreadPoolExecutor
import os
from joblib import cpu_count
import fd
import logging
import pickle
from itertools import combinations

import pandas as pd
import ud_utils as udt

def fd_process_col(path: str, fd_pair: tuple, train_df: pd.DataFrame, tokens_dict: dict):
    """"
    This function computes the functional dependencies for a pair of columns in a table.
    parameters:
    ----------
    :param path: path to the table
    :param fd_pair: pair of columns
    :param train_df: pandas dataframe
    :param tokens_dict: tokens dictionary
    :return: dictionary of functional dependencies for the pair of columns
    """
    cols_id = path + "_" + fd_pair[0] + fd_pair[1]
    col_measures = fd.get_col_measures(train_df[fd_pair[0]], train_df[fd_pair[1]],
                                            list(train_df.columns).index(fd_pair[0]),
                                            list(train_df.columns).index(fd_pair[1]), tokens_dict)
    return cols_id, col_measures

def fd_process_table(path: str, output_path: str, file_type: str, tokens_dict: dict, executor: ThreadPoolExecutor) -> dict:
    """
    This function computes the functional dependencies for a table.
    parameters:
    ----------
    :param path: path to the table
    :param output_path: path to save the dictionary of functional dependencies
    :param file_type: type of the file
    :param tokens_dict: tokens dictionary
    :param executor: ThreadPoolExecutor
    :return: dictionary of functional dependencies for the table
    """
    try:
        if file_type == "parquet":
            train_df = pd.read_parquet(path)
        else:
            train_df = pd.read_csv(path)

        executor_features = []
        # functional dependencies
        for pair in combinations(train_df.columns, 2):
            if pair[0] != pair[1]:
                executor_features.append(executor.submit(fd_process_col, path, pair, train_df, tokens_dict))
        path_fd_dict = {}
        for feature in executor_features:
            col_id, col_measures = feature.result()
            if col_measures is not None:
                path_fd_dict[col_id] = col_measures
        # Save the dictionary for the table to disk
        with open(output_path + "/" + os.path.basename(path).removesuffix("." + file_type) + ".pickle", 'wb') as f:
            pickle.dump(path_fd_dict, f)
        logging.info(f"Finish df: {path}, df shape: {train_df.shape}")
    except Exception as e:
        logging.info(f"Error {e} processing path {path}")

def fd_offline_learning(train: list, file_type: str, output_path: str, tokens_dict: dict) -> dict:
    """
    This function computes the functional dependencies for each table in the training set in parallel.
    parameters:
    ----------
    :param train: list of paths to the training data
    :param file_type: parquet or csv
    :param output_path: path to save the results
    :param tokens_dict: tokens dictionary
    :return: dictionary of functional dependencies
    """

    fd_dict = {}
    tables_output_path = os.path.join(output_path, "fd_tables_results")
    if not os.path.exists(tables_output_path):
        os.makedirs(tables_output_path)
    with ThreadPoolExecutor(max_workers=cpu_count() * 2) as executor:
        executor_features = []
        for path in train:
            executor_features.append(executor.submit(fd_process_table, path, tables_output_path, file_type, tokens_dict, executor))
        for feature in executor_features:
            table_fd_dict = feature.result()
            if table_fd_dict is not None:
                fd_dict.update(table_fd_dict)
    logging.info(f"Writing fd_dict to {output_path}")
    with open(os.path.join(output_path, "fd_dict.pickle"), 'wb') as f:
        pickle.dump(fd_dict, f)
    return fd_dict