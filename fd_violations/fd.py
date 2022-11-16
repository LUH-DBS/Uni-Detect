from uv import uniqueness as uv
import ud_utils as udt
import numpy as np


def get_fr(column_1, column_2):
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


def get_col_measures(col_1, col_2, left_ness_col_1, left_ness_col_2, tokens_dict):
    cols_perturbed = perturbation(col_1, col_2)
    col_dict = {"d_type_1": col_1.dtype,
                "d_type_2": col_2.dtype,
                "number_of_rows_range": udt.get_range_count(col_1.count()),
                "left_ness_1": left_ness_col_1,
                "left_ness_2": left_ness_col_2,
                "avg_col_pre_1": uv.get_prev_range(tokens_dict, col_1),
                "avg_col_pre_2": uv.get_prev_range(tokens_dict, col_2),
                "fd": cols_perturbed[0] if cols_perturbed else np.nan,
                "fd_p": cols_perturbed[1] if cols_perturbed else np.nan
                }
    return col_dict


def perturbation(column_1, column_2):
    fr_d, idx_d = get_fr(column_1, column_2)
    if idx_d == -1:
        return fr_d, fr_d, idx_d
    p_column_1, p_column_2 = column_1.drop(idx_d), column_2.drop(idx_d)
    p_column_1 = p_column_1.reset_index(drop=True)
    p_column_2 = p_column_2.reset_index(drop=True)
    fr_do, tmp = get_fr(p_column_1, p_column_2)

    return fr_d, fr_do, idx_d
