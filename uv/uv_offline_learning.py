
from concurrent.futures import ThreadPoolExecutor
import os
import sys

from joblib import cpu_count
import uniqueness as uv
import pandas as pd
import logging
import pickle

sys.path.append(os.path.abspath(os.path.join('.')))
import ud_utils as udt

def uv_process_col(path: str, col_name: str, train_df: pd.DataFrame, tokens_dict: dict) -> tuple[str, dict]:
    """
    Run uniqueness offline learning for a single column

    Parameters
    ----------
    :param path: path to the train table
    :param col_name: name of the column to process
    :param train_df: train dataframe
    :return: tuple with the column id and the uniqueness measures
    """
    col_id = path + "_" + col_name
    col = train_df[col_name]
    col_measures = None
    # Ignore empty cols
    if not col.empty:
        # Get column measures using the provided function
        col_measures = uv.get_col_measures(col, list(train_df.columns).index(col_name), tokens_dict)
    logging.info(f"Finish col_id: {col_id} df: {path}")
    return col_id, col_measures


def uv_process_table(path: str, output_path: str, file_type: str, tokens_dict: dict, executor: ThreadPoolExecutor) -> dict:
    """
    Run uniqueness offline learning for a single table

    Parameters
    ----------
    :param path: path to the train table
    :param output_path: path to save the dictionary
    :param file_type: file type of the train table
    :param executor: executor to run the offline learning in parallel
    :param tokens_dict: dictionary with the tokens
    :return: dictionary with the uniqueness features
    """
    logging.info(f"Start df: {path}")
    try:
        if file_type == "parquet":
            train_df = pd.read_parquet(path + "/clean.parquet")
        else:
            train_df = pd.read_csv(path)

        executor_features = []
        for col_name in train_df.columns:
            executor_features.append(executor.submit(uv_process_col, path, col_name, train_df, tokens_dict))
        path_uv_dict = {}
        for feature in executor_features:
            col_id, col_measures = feature.result()
            if col_measures is not None:
                path_uv_dict[col_id] = col_measures
        # Save the dictionary for the table to disk
        with open(output_path + "/" + path.split("/")[-1] + ".pickle", 'wb') as f:
            pickle.dump(path_uv_dict, f)
        logging.info(f"Finish df: {path}, df shape: {train_df.shape}")
        return path_uv_dict
    except Exception as e:
        logging.error(f"Error: {e}")
        logging.info(f"Error processing df: {path}, df shape: {train_df.shape}")
    return  {}

def uv_offline_learning(train_path_list: list, file_type: str, output_path: str):
    """
    Run uniqueness offline learning for a list of tables

    Parameters
    ----------
    :param train_path_list: list of paths to the train tables
    :param file_type: file type of the train tables
    :param output_path: path to save the dictionary
    :return: dictionary with the uniqueness features
    """
    uv_dict = {}
    tables_output_path = os.path.join(output_path, "uv_tables_results")
    if not os.path.exists(tables_output_path):
        os.makedirs(tables_output_path)
    with ThreadPoolExecutor(max_workers=cpu_count() * 2) as executor:
        tokens_dict = uv.get_tokens_dict(train_path_list, output_path, file_type, executor)
        executor_features = []
        for path in train_path_list:
            executor_features.append(executor.submit(uv_process_table, path, tables_output_path, file_type, tokens_dict, executor))
        for feature in executor_features:
            path_dict = feature.result()
            if path_dict is not None:
                uv_dict.update(path_dict)
    logging.info(f"Writting uv_dict")
    with open(os.path.join(output_path, "uv_dict.pickle"), 'wb') as f:
        pickle.dump(uv_dict, f)
