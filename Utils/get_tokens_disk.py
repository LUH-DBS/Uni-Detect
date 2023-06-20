import os
import logging
import pickle
from collections import Counter
import itertools
import time

def load_tokens(tokens_dir_path):
    for path in os.listdir(tokens_dir_path):
        with open(os.path.join(tokens_dir_path, path), 'rb') as f:
            yield pickle.load(f)

def aggregate_tokens(tokens):
    aggregated_tokens_counter = Counter()
    n_tables = 0
    len_tokens_dicts = 0
    for token_dict in tokens:
        aggregated_tokens_counter.update(token_dict)
        n_tables += 1
        len_tokens_dicts += len(token_dict)
    return aggregated_tokens_counter, n_tables, len_tokens_dicts

def chunks(iterable, size):
    iterator = iter(iterable)
    for first in iterator:
        yield list(itertools.chain([first], itertools.islice(iterator, size - 1)))

def write_tokens_dict(tokens_dict, output_path, chunk_size=10000000):
    i = 0
    for chunk in chunks(tokens_dict.items(), chunk_size):
        with open(os.path.join(output_path, f'tokens_dict_{i}_{time.time()}.pkl'), 'wb') as f:
            pickle.dump(dict(chunk), f)
        i += 1

def get_dict(tokens_dir_path, output_path):
    tokens = load_tokens(tokens_dir_path)

    aggregated_tokens_counter, n_tables, len_tokes_dicts = aggregate_tokens(tokens)
    print("********n_tables********")
    print(n_tables)
    logging.info("Finish aggregating tokens dict")

    print("********len_tokes_dicts********")
    print(len_tokes_dicts)

    tokens_dict = {}
    for key, value in aggregated_tokens_counter.items():
        tokens_dict[key] = value / n_tables
    logging.info("Finish getting tokens dict")

    write_tokens_dict(tokens_dict, output_path)
    return tokens_dict

if __name__ == '__main__':
    get_dict('/home/fatemeh/EDS-BaseLines/Uni-Detect/Utils/tokens_dir', '/home/fatemeh/EDS-BaseLines/Uni-Detect/Utils/tokens-dict-3')
