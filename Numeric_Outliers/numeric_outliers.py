import numpy
from scipy import stats
import numpy as np
import ud_utils as udt


def get_max_mad_score(column):
    max_mad, max_idx = -np.inf, -1
    mad = stats.median_abs_deviation(column.to_numpy())
    median = column.median()
    for idx, value in column.items():
        mad_score = -np.inf
        if not numpy.isnan(mad) and mad != 0:
            mad_score = abs(value - median) / mad
        if mad_score > max_mad:
            max_mad = mad_score
            max_idx = idx
    return max_mad, max_idx


def get_col_measures(col):
    col_perturbed = perturbation(col)
    col_dict = {"d_type": col.dtype,
                "number_of_rows_range": udt.get_range_count(col.count()),
                "max_mad": col_perturbed[0] if col_perturbed else np.nan,
                "max_mad_p": col_perturbed[1] if col_perturbed else np.nan,
                # TODO it's not memory efficient
                "col_transformed": np.log(col)
    }
    return col_dict


def perturbation(column):
    max_mad_d, max_idx = get_max_mad_score(column)
    if max_idx == -1:
        return
    p_column = column.drop(max_idx)
    p_column = p_column.reset_index(drop=True)
    max_mad_do, temp = get_max_mad_score(p_column)
    return max_mad_d, max_mad_do, max_idx

