from nltk import word_tokenize
import numpy as np
from Levenshtein import distance
from scipy.spatial.distance import pdist, squareform
import ud_utils as udt


def get_avg_diff_tokens(value_1, value_2):
    tokenized_value_1 = word_tokenize(value_1)
    tokenized_value_2 = word_tokenize(value_2)
    uniq_1 = list(set(tokenized_value_1) - set(tokenized_value_2))
    uniq_2 = list(set(tokenized_value_2) - set(tokenized_value_1))
    uniq = uniq_1 + uniq_2
    if not uniq or len(uniq) == 0:
        return 0
    count_char = 0
    for token in uniq:
        count_char += len(token)
    return count_char / len(uniq)


# mpd: minimum pairwise distance
def get_mpd(column):
    transformed_col = column.to_numpy().reshape(-1, 1)
    try:
        distance_matrix = pdist(transformed_col, lambda x, y: distance(x[0], y[0]))
    except Exception as e:
        print(e)
        return
    sdm = squareform(distance_matrix)
    np.nan_to_num(sdm, np.inf)
    sdm[sdm == 0] = np.inf
    mpd = np.nanmin(sdm)
    idx = np.unravel_index(int(sdm.argmin()), sdm.shape)
    if idx:
        i_p, j_p = idx[0], idx[1]
    else:
        i_p, j_p = -1, -1
    avg_diff_tokens = get_avg_diff_tokens(column.loc[i_p], column.loc[j_p])
    avg_len_diff_tokens = udt.get_range_mpd(avg_diff_tokens) if avg_diff_tokens else -1
    return mpd, i_p, j_p, avg_len_diff_tokens

def get_col_measures(col):
    col_perturbed = perturbation(col)
    col_dict = {"d_type": col.dtype,
                "number_of_rows_range": udt.get_range_count(col.count()),
                "range_mpd": col_perturbed[2] if col_perturbed else np.nan,
                "mpd": col_perturbed[0] if col_perturbed else np.nan,
                "mpd_p": col_perturbed[1] if col_perturbed else np.nan}
    return col_dict


def perturbation(column):
    column = column.astype(str)
    mpd_d, i_p, j_p, avg_len_diff_tokens = get_mpd(column)
    if i_p == -1 or j_p == -1 or avg_len_diff_tokens == -1 or mpd_d == np.inf:
        return
    p_column_test_1 = column.drop(i_p)
    p_column_test_1 = p_column_test_1.reset_index(drop=True)
    mpd_do_test_1, i_p_test_1, j_p_test_1, tmp_test_1 = get_mpd(p_column_test_1)

    p_column_test_2 = column.drop(j_p)
    p_column_test_2 = p_column_test_2.reset_index(drop=True)
    mpd_do_test_2, i_p_test_2, j_p_test_2, tmp_test_2 = get_mpd(p_column_test_2)

    if mpd_do_test_2 > mpd_do_test_1:
        mpd_do, idx = mpd_do_test_2, j_p
    else:
        mpd_do, idx = mpd_do_test_1, i_p
    return mpd_d, mpd_do, avg_len_diff_tokens, idx

