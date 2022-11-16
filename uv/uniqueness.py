import ud_utils as udt
from nltk import word_tokenize
import numpy as np
import pandas as pd


def offline_learning_corpus(train_path):
    tokens_dict = {}
    tokens_sum_dict = {}
    tokens_set = set()
    n_tables = len(train_path)
    for path in train_path:
        train_df = pd.read_parquet(path + "/clean.parquet")
        for col_name in train_df.columns:
            train_df[col_name] = train_df[col_name].astype(str)
            for idx, value in train_df[col_name].items():
                tokens = word_tokenize(value)
                for token in tokens:
                    tokens_set.add(token)
    for token in tokens_set:
        if token in tokens_sum_dict.keys():
            tokens_sum_dict[token] += 1
        else:
            tokens_sum_dict[token] = 1
    for token in tokens_sum_dict.keys():
        tokens_dict[token] = tokens_sum_dict[token] / n_tables
    return tokens_dict


def get_prev_range(tokens_dict, col):
    tokens_set_col = set()
    prev_sum = -1
    col = col.astype(str)
    for idx, value in col.items():
        tokens = word_tokenize(value)
        for token in tokens:
            tokens_set_col.add(token)
    for token in tokens_set_col:
        if token in tokens_dict.keys():
            prev_sum += tokens_dict[token]
        else:
            prev_sum = 1
    prev_avg = prev_sum / len(tokens_set_col)
    prev_avg_range = udt.get_range_avg_pre(prev_avg)
    return prev_avg_range


def get_uniqueness(column):
    uniqueness, dup_index = column.nunique() / column.count(), -1

    for i, value_1 in column.items():
        for j, value_2 in column.items():
            if j > i and value_1 == value_2:
                return uniqueness, i
    return uniqueness, dup_index


def get_col_measures(col, left_ness, tokens_dict):
    col_perturbed = perturbation(col)
    col_dict = {"d_type": col.dtype,
                "number_of_rows_range": udt.get_range_count(col.count()),
                "left_ness": left_ness,
                "avg_col_pre": get_prev_range(tokens_dict, col),
                "ur": col_perturbed[0] if col_perturbed else np.nan,
                "ur_p": col_perturbed[1] if col_perturbed else np.nan
                }
    return col_dict


def perturbation(column):
    uniqueness_d, i_duplicate = get_uniqueness(column)
    if i_duplicate == -1:
        return uniqueness_d, uniqueness_d, i_duplicate
    p_column = column.drop(i_duplicate)
    p_column = p_column.reset_index(drop=True)
    uniqueness_do, tmp = get_uniqueness(p_column)

    return uniqueness_d, uniqueness_do, i_duplicate
