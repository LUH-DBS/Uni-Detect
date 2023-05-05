from concurrent.futures import ThreadPoolExecutor
import logging
import sys
import os
sys.path.append(os.path.abspath(os.path.join('.')))
import ud_utils as udt
from nltk import word_tokenize
import numpy as np
import pandas as pd
import pickle

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
    logging.info(f"Start getting measures for column {col.name}")
    col_perturbed = perturbation(col)
    str_col = col.astype(str)
    col_dict = {"d_type": "alnumeric" if str_col.str.isalnum().all() else col.dtype,
                "number_of_rows_range": udt.get_range_count(col.count()),
                "left_ness": left_ness,
                "avg_col_pre": udt.get_prev_range(tokens_dict, col),
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
