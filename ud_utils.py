from concurrent.futures import ThreadPoolExecutor
import logging
import os
import pickle
from nltk import word_tokenize
import pandas as pd

def get_range_count(num_rows: int) -> int:
    """
    Get the range of the number of rows
    
    parametrers:
    ------------
    :param num_rows: number of rows
    :return: range of the number of rows
    """
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

def get_range_mpd(mpd_diff: float) -> int:
    """
    Get the range of the average length of the tokens that difer between the MPD pair
    
    parametrers:
    ------------
    :param mpd_diff: minimum pairwise distance
    :return: range of the average length of the tokens that difer between the MPD pair
    """
    if mpd_diff <= 5:
        return 0
    if mpd_diff <= 10:
        return 1
    if mpd_diff <= 15:
        return 2
    if mpd_diff <= 20:
        return 3
    return 4

def get_range_avg_pre(avg_tokens: float) -> int:
    """
    Get the range of the average number of tokens pervalance

    parametrers:
    ------------
    :param avg_tokens: average number of tokens pervalance
    :return: range of the average number of tokens pervalance
    """
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

def get_tokens_dict(train_path: str, output_path: str, file_type: str, executor: ThreadPoolExecutor):
    """
    This function calculates the tokens dictionary. Tokens_dict is a dictionary that maps each token to its frequency / number of tables.
    parameters
    ----------
    :param train_path: str
        The path to the train data.
    :param output_path: str
        The path to save the tokens dictionary.
    :param file_type: str
        The file type of the train data.
    :param executor: ThreadPoolExecutor
        The executor to use for parallelization.
    :return: dict
        The tokens dictionary.
    """
    logging.info(f"Start getting tokens dict")
    tokens_dict = {}
    tokens_list = []
    tokens_set = set()
    n_tables = len(train_path)
    executor_features = []

    for path in train_path:
        executor_features.append(executor.submit(get_table_tokens_dict, path, file_type))
    for feature in executor_features:
        tokens_list.extend(feature.result())
    tokens_set = set(tokens_list)
    token_counts = {token: tokens_list.count(token) for token in tokens_set}
    tokens_dict = {token: token_counts[token] / n_tables for token in token_counts}
    with open(os.path.join(output_path, 'tokens_dict.pkl'), 'wb') as f:
        pickle.dump(tokens_dict, f)
    return tokens_dict

def get_table_tokens_dict(table_path: str, file_type: str) -> set:
    """
    This function calculates the tokens dictionary for a single table.
    parameters
    ----------
    :param table_pat: str
        The path to the table.
    :param file_type: str
        The file type of the table.
    :return: set
        The tokens dictionary for the table.
    """
    logging.info(f"Start getting tokens dict for table {table_path}")
    tokens_list = []
    if file_type == "parquet":
        train_df = pd.read_parquet(table_path)
    else:
        train_df = pd.read_csv(table_path)
    for col_name in train_df.columns:
        train_df[col_name] = train_df[col_name].astype(str)
        for idx, value in train_df[col_name].items():
            tokens = word_tokenize(value)
            for token in tokens:
                tokens_list.append(token)
    return tokens_list

def get_prev_range(tokens_dict: dict, col: pd.Series) -> float:
    """
    This function calculates average prevalenve of the column.
    parameters
    ----------
    :param tokens_dict: dict
        The dictionary of the tokens.
    :param col: pd.Series
        The column to calculate the average prevalence for.
    :return: float 
        The average prevalence of the column.
    """
    tokens_set_col = set()
    prev_sum = -1
    col = col.astype(str)

    tokens_list_col = [token for idx, value in col.items() for token in word_tokenize(value)]
    tokens_set_col = set(tokens_list_col)
    prev_sum = sum(tokens_dict.get(token, 1) for token in tokens_set_col)
    prev_avg = prev_sum / len(tokens_set_col)
    prev_avg_range = get_range_avg_pre(prev_avg)
    return prev_avg_range