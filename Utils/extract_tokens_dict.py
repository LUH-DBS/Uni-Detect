

from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import logging
import os
import pickle
import sys
import time
from joblib import cpu_count
import pandas as pd
import yaml


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
    n_tables = len(train_path)
    executor_features = []

    for path in train_path:
        executor_features.append(executor.submit(get_table_tokens_dict, path, file_type))
    td = [feature.result() for feature in executor_features]
    aggregated_tokens_counter = sum((Counter(token_dict) for token_dict in td), Counter())
    tokens_dict = {k: v / n_tables for k, v in aggregated_tokens_counter.items()}
    logging.info(f"Finish getting tokens dict")
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
    tokens_dict = {}
    if file_type == "parquet":
        train_df = pd.read_parquet(table_path)
    else:
        train_df = pd.read_csv(table_path).astype(str)

    # concatenate all the columns into a single Series
    text_series = train_df.apply(lambda x: ' '.join(x.astype(str)), axis=1)
    # tokenize the text in each row of the Series and concatenate the resulting Series
    tokens = text_series.str.split().explode()
    # count the frequency of each token
    token_counts = tokens.value_counts()
    # create a dictionary with tokens as keys and their frequency as values
    tokens_dict = token_counts.to_dict()
    logging.info(f"Finish getting tokens dict for table {table_path}")
    return tokens_dict

if __name__ == "__main__":
    t0 = time.time()
    # Load config file
    with open(sys.argv[1]) as config_file:
        config = yaml.load(config_file, Loader=yaml.SafeLoader)

    # Set up logging
    logging.basicConfig(filename=config['log_path'] + '_app.log', filemode='w',
                        format='%(name)s - %(levelname)s - %(message)s', level=logging.INFO)

    train_path_list = config['train_path_list']
    output_path = config['output_path']
    file_type = config['file_type']

    with open(train_path_list, 'rb') as f:
        train_path_list = pickle.load(f)
    with ThreadPoolExecutor(max_workers=cpu_count() * 2) as executor:
        tokens_dict = get_tokens_dict(train_path_list, output_path, file_type, executor)
    logging.info(f"Total time: {time.time() - t0}")
    
            