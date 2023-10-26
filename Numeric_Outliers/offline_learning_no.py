import logging
import os
import pickle
from concurrent.futures import Executor, ThreadPoolExecutor

import numeric_outliers as no
import numpy as np
import pandas as pd
from joblib import cpu_count


def no_process_col(
    path: str, col_name: str, train_df: pd.DataFrame
) -> tuple[str, dict]:
    """
    Run numeric outliers offline learning for a single column
    :param path: path to the train table
    :param col_name: name of the column to process
    :param train_df: train dataframe
    :return: tuple with the column id and the numeric outliers measures
    """
    try:
        col_id = path + "_" + col_name
        col = train_df[col_name]
        # Ignore empty cols
        if not col.empty:
            # Get column measures
            col_measures = no.get_col_measures(col)
            return col_id, col_measures
        return col_id, None
    except Exception as e:
        logging.error(f"Error processing column {col_name}: {e}")
        return col_id, None


# Define the function to be applied to each path
def no_process_table(
    path: str, output_path: str, file_type: str, executor: Executor, n_cells_limit: int
) -> dict:
    """
    Run numeric outliers offline learning for a single table
    :param path: path to the train table
    :param file_type: file type of the train table
    :param executor: executor to run the offline learning in parallel
    :return: dictionary with the numeric outliers features
    """
    try:
        logging.info(f"Processing path {path}")
        if file_type == "parquet":
            train_df = pd.read_parquet(path)
        else:
            train_df = pd.read_csv(path)
        if train_df.shape[0] * train_df.shape[1] < n_cells_limit:
            # Get only numeric columns
            train_df_no = train_df.select_dtypes(include=[np.number])
            path_no_dict = {}
            executor_features = []
            for col_name in train_df_no.columns:
                executor_features.append(
                    executor.submit(no_process_col, path, col_name, train_df_no)
                )

            for feature in executor_features:
                col_id, col_measures = feature.result()
                path_no_dict[col_id] = col_measures

            logging.info(
                f"Processed path {path}, df shape: {train_df.shape}, df_no shape: {train_df_no.shape}"
            )
        else:
            logging.info(f"Skipping path {path}, df shape: {train_df.shape}")
            path_no_dict = {}
        # Save the dictionary for the table to disk
        with open(
            output_path
            + "/"
            + os.path.basename(path).removesuffix("." + file_type)
            + ".pickle",
            "wb",
        ) as f:
            pickle.dump(path_no_dict, f)
        return path_no_dict

    except Exception as e:
        logging.error(f"Error processing path {path}: {e}")
        return {}


def no_offline_learning(
    train_path_list: list, file_type: str, output_path: str, n_cells_limit: int
) -> dict:
    """
    Run numeric outliers offline learning
    :param train_path_list: list of paths to train tables
    :param file_type: file type of the train tables
    :param output_path: path to save the dictionary
    :return: dictionary with the numeric outliers features
    """
    no_dict = {}
    tables_output_path = os.path.join(output_path, "no_tables_results")
    if not os.path.exists(tables_output_path):
        os.makedirs(tables_output_path)
    # Run offline learning in parallel using all the cores available
    with ThreadPoolExecutor(max_workers=cpu_count() * 2) as executor:
        for path in train_path_list:
            path_dict = no_process_table(
                path, tables_output_path, file_type, executor, n_cells_limit
            )
            no_dict.update(path_dict)

    # Save the dictionary to disk
    with open(os.path.join(output_path, "no_dict.pickle"), "wb") as f:
        pickle.dump(no_dict, f)
    return no_dict
