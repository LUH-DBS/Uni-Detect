
from concurrent.futures import Executor, ThreadPoolExecutor
import logging
from multiprocessing import Pool
import pickle
from joblib import cpu_count
import numpy as np
import pandas as pd
import numeric_outliers as no

def no_process_col(path, col_name, train_df):
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
def no_process_path(path, file_type, executor):
    try:
        if file_type == "parquet":
            train_df = pd.read_parquet(path + "/clean.parquet")
        else:
            train_df = pd.read_csv(path)
        
        # numeric outliers
        train_df_no = train_df.select_dtypes(include=[np.number])
        path_no_dict = {}
        executor_features = []
        for col_name in train_df_no.columns:
            executor_features.append(executor.submit(no_process_col, path, col_name, train_df_no))

        for feature in executor_features:
            col_id, col_measures = feature.result()
            path_no_dict[col_id] = col_measures

        logging.info(f"df shape: {train_df.shape}")
        return path_no_dict
    except Exception as e:
        logging.error(f"Error processing path {path}: {e}")
        return {}
        
def no_offline_learning(train, file_type, output_path):
    no_dict = {}

    with ThreadPoolExecutor(max_workers=cpu_count() * 2) as executor:
        executor_features = []
        for path in train:
            executor_features.append(executor.submit(no_process_path, path, file_type, executor))
        for feature in executor_features:
            path_dict = feature.result()
            no_dict.update(path_dict)
    
    # Save the dictionary to disk
    with open(output_path, 'wb') as f:
        pickle.dump(no_dict, f)
    return no_dict