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