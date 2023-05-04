from concurrent.futures import ThreadPoolExecutor
import sys
import os
sys.path.append(os.path.abspath(os.path.join('.')))
import ud_utils as udt
from nltk import word_tokenize
import numpy as np
import pandas as pd
import pickle

def get_table_tokens_dict(table_pat: str, file_type: str) -> set:
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
    tokens_list = []
    if file_type == "parquet":
        train_df = pd.read_parquet(table_pat + "/clean.parquet")
    else:
        train_df = pd.read_csv(table_pat)
    for col_name in train_df.columns:
        train_df[col_name] = train_df[col_name].astype(str)
        for idx, value in train_df[col_name].items():
            tokens = word_tokenize(value)
            for token in tokens:
                tokens_list.append(token)
    return tokens_list

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
    prev_sum = sum(tokens_dict.get(token, 1) - tokens_list_col.count(token) for token in tokens_set_col)
    prev_avg = prev_sum / len(tokens_set_col)
    prev_avg_range = udt.get_range_avg_pre(prev_avg)
    return prev_avg_range


def get_uniqueness(column: pd.Series) -> tuple[float, int]:
    """
    This function calculates the uniqueness of a column.
    parameters
    ----------
    :param column: pd.Series
        The column to calculate the uniqueness for.
    :return: tuple[float, int]
    """
    uniqueness, dup_index = column.nunique() / column.count(), -1

    for i, value_1 in column.items():
        for j, value_2 in column.items():
            if j > i and value_1 == value_2:
                return uniqueness, i
    return uniqueness, dup_index


def get_col_measures(col: pd.Series, left_ness: int, tokens_dict: dict) -> dict:
    """
    This function calculates the measures of a column.
    parameters
    ----------
    :param col: pd.Series
        The column to calculate the measures for.
    :param left_ness: int
        The leftness of the column.
    :param tokens_dict: dict
        The dictionary of the tokens.
    :return: dict   
    """
    col_perturbed = perturbation(col)
    str_col = col.astype(str)
    col_dict = {"d_type": "alnumeric" if str_col.str.isalnum().all() else col.dtype,
                "number_of_rows_range": udt.get_range_count(col.count()),
                "left_ness": left_ness,
                "avg_col_pre": get_prev_range(tokens_dict, col),
                "ur": col_perturbed[0] if col_perturbed else np.nan,
                "ur_p": col_perturbed[1] if col_perturbed else np.nan
                }
    return col_dict


def perturbation(column: pd.Series) -> tuple[float, float, int]:
    """
    This function perturbs the column and returns the uniqueness of the original column, the uniqueness of the
    perturbed column, and the index of the duplicate value in the original column.
    
    Parameters
    ----------
    :param: column : pd.Series
        The column to be perturbed.
    :return: tuple[float, float, int]
        The uniqueness of the original column, the uniqueness of the perturbed column, and the index of the duplicate
    """
    uniqueness_d, i_duplicate = get_uniqueness(column)
    if i_duplicate == -1:
        return uniqueness_d, uniqueness_d, i_duplicate
    p_column = column.drop(i_duplicate)
    p_column = p_column.reset_index(drop=True)
    uniqueness_do, tmp = get_uniqueness(p_column)

    return uniqueness_d, uniqueness_do, i_duplicate
