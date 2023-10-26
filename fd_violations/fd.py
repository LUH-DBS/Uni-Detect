import os
import sys

import pandas as pd

sys.path.append(os.path.abspath(os.path.join(".")))
import numpy as np

import ud_utils as udt


def get_fr(column_1: pd.Series, column_2: pd.Series) -> tuple[float, int]:
    """
    Get the FD compliance ratio and the index of the first violation

    parametrers:
    ------------
    :param column_1: first column
    :param column_2: second column
    :return: FD compliance ratio and the index of the first violation
    """
    dict_values = {}
    violation_keys = set()
    p_idx = -1
    for idx in range(column_1.count()):
        val_1, val_2 = column_1.at[idx], column_2.at[idx]
        if val_1 in violation_keys:
            continue
        if val_1 in dict_values.keys() and val_2 != dict_values.get(val_1):
            violation_keys.add(val_1)
            dict_values.pop(val_1)
            if p_idx == -1:
                p_idx = idx
        else:
            dict_values[val_1] = val_2
    if not dict_values:
        return -1, -1
    fr = len(dict_values) / column_1.count()
    return fr, p_idx


def get_col_measures(
    col_1: pd.Series,
    col_2: pd.Series,
    left_ness_col_1: int,
    left_ness_col_2: int,
    tokens_dict: dict,
) -> dict:
    """
    Get the measures of the columns

    parametrers:
    ------------
    :param col_1: first column
    :param col_2: second column
    :param left_ness_col_1: left_ness of the first column
    :param left_ness_col_2: left_ness of the second column
    :param tokens_dict: tokens dictionary
    :return: measures of the columns
    """
    cols_perturbed = perturbation(col_1, col_2)
    str_col_1 = col_1.astype(str)
    str_col_2 = col_2.astype(str)
    col_dict = {
        "d_type_1": "alnumeric" if str_col_1.str.isalnum().all() else col_1.dtype,
        "d_type_2": "alnumeric" if str_col_2.str.isalnum().all() else col_2.dtype,
        "number_of_rows_range": udt.get_range_count(col_1.count()),
        "left_ness_1": left_ness_col_1,
        "left_ness_2": left_ness_col_2,
        "avg_col_pre_1": udt.get_prev_range(tokens_dict, col_1),
        "avg_col_pre_2": udt.get_prev_range(tokens_dict, col_2),
        "fd": cols_perturbed[0] if cols_perturbed else np.nan,
        "fd_p": cols_perturbed[1] if cols_perturbed else np.nan,
    }
    return col_dict


def perturbation(column_1: pd.Series, column_2: pd.Series) -> tuple[float, float, int]:
    """
    Perturbation function

    parametrers:
    ------------
    :param column_1: first column
    :param column_2: second column
    :return: perturbation results and index of the first violation
    """
    fr_d, idx_d = get_fr(column_1, column_2)
    if idx_d == -1:
        return fr_d, fr_d, idx_d
    p_column_1, p_column_2 = column_1.drop(idx_d), column_2.drop(idx_d)
    p_column_1 = p_column_1.reset_index(drop=True)
    p_column_2 = p_column_2.reset_index(drop=True)
    fr_do, tmp = get_fr(p_column_1, p_column_2)
    return fr_d, fr_do, idx_d
