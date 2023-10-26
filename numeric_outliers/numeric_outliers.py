import os
import sys

import numpy
import numpy as np
import pandas as pd
from scipy import stats

sys.path.append(os.path.abspath(os.path.join(".")))
import ud_utils as udt


def get_max_mad_score(column: pd.Series) -> tuple:
    """
    Get the maximum MAD score for a column
    :param column: column to process
    :return: tuple with the max MAD score and the index of the row with the max MAD score
    """
    max_mad, max_idx = -np.inf, -1
    mad = stats.median_abs_deviation(column.to_numpy(), nan_policy="omit")
    median = column.median()
    for idx, value in column.items():
        mad_score = -np.inf
        if not numpy.isnan(mad) and mad != 0:
            mad_score = abs(value - median) / mad
        if mad_score > max_mad:
            max_mad = mad_score
            max_idx = idx
    return max_mad, max_idx


def get_col_measures(col: pd.Series) -> dict:
    """
    Get the numeric outliers measures for a column
    :param col: column to process
    :return: dictionary with the numeric outliers measures
    """
    col_perturbed = perturbation(col)
    col_dict = {
        "d_type": col.dtype,
        "number_of_rows_range": udt.get_range_count(col.count()),
        "max_mad": col_perturbed[0] if col_perturbed else np.nan,
        "max_mad_p": col_perturbed[1] if col_perturbed else np.nan,
        # TODO it's not memory efficient
        "col_transformed": np.log(col),
    }
    return col_dict


def perturbation(column: pd.Series) -> tuple[float, float, int]:
    """
    Perturb the column by removing the row with the highest MAD score and calculate the MAD score again
    :param column: column to perturb
    :return: max MAD score, max MAD score after perturbation, index of the row with the max MAD score
    """

    max_mad_d, max_idx = get_max_mad_score(column)
    if max_idx == -1:
        return None, None, None
    p_column = column.drop(max_idx)
    p_column = p_column.reset_index(drop=True)
    max_mad_do, temp = get_max_mad_score(p_column)
    return max_mad_d, max_mad_do, max_idx
