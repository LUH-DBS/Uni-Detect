
from concurrent.futures import ThreadPoolExecutor
import logging
import os
import pickle

import pandas as pd
import spelling_errors as se
from joblib import cpu_count


def se_process_col(path: str, col_name: str, train_df: pd.DataFrame) -> tuple[str, dict]:
    """
    Run spelling errors offline learning for a single column
    :param path: path to the train table
    :param col_name: name of the column to process
    :param train_df: train dataframe
    :return: tuple with the column id and the spelling errors measures
    """
    col_id = path + "_" + col_name
    col = train_df[col_name]
    col_measures = None
    # Ignore empty cols
    if not col.empty:
        # Get column measures using the provided function
        col_measures = se.get_col_measures(col)
    logging.info(f"Finish col_id: {col_id} df: {path}")
    return col_id, col_measures


def se_process_table(path: str, output_path: str, file_type: str, executor: ThreadPoolExecutor) -> dict:
    """
    Run spelling errors offline learning for a single table
    :param path: path to the train table
    :param file_type: file type of the train table
    :param executor: executor to run the offline learning in parallel
    :return: dictionary with the spelling errors features
    """
    try:
        if file_type == "parquet":
            train_df = pd.read_parquet(path + "/clean.parquet")
        else:
            train_df = pd.read_csv(path)
        
        executor_features = []
        for col_name in train_df.columns:
            executor_features.append(executor.submit(se_process_col, path, col_name, train_df))

        path_se_dict = {}
        for feature in executor_features:
            col_id, col_measures = feature.result()
            if col_measures is not None:
                path_se_dict[col_id] = col_measures
        # Save the dictionary for the table to disk
        with open(output_path + "/" + path.split("/")[-1] + ".pickle", 'wb') as f:
            pickle.dump(path_se_dict, f)
        logging.info(f"Finish df: {path}, df shape: {train_df.shape}")
        return path_se_dict
    except Exception as e:
        logging.error(e)
        logging.info(f"Error processing path {path}")
        return {}

def se_offline_learning(train_path_list: list, file_type: str, output_path:str) -> dict:
    """
    Offline learning for spelling errors
    train_path_list: list of paths to train datasets
    file_type: "csv" or "parquet"
    output_path: path to save the output
    """
    se_dict = {}
    tables_output_path = os.path.join(output_path, "se_tables_results")
    if not os.path.exists(tables_output_path):
        os.makedirs(tables_output_path)
    with ThreadPoolExecutor(max_workers=cpu_count() * 2) as executor:
        executor_features = []
        for path in train_path_list:
            executor_features.append(executor.submit(se_process_table, path, tables_output_path, file_type, executor))
        for feature in executor_features:
            path_dict = feature.result()
            if path_dict is not None:
                se_dict.update(path_dict)
    logging.info(f"Writting se_dict")
    with open(os.path.join(output_path, "se_dict.pickle"), 'wb') as f:
        pickle.dump(se_dict, f)
    return se_dict