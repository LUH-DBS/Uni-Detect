from multiprocessing import Pool
import os
import logging
import pickle
from collections import Counter

def count_tokens(token_dict):
    return Counter(token_dict)


def get_dict(tokens_dir_path, output_path):

    with open("/home/fatemeh/EDS-BaseLines/Uni-Detect/output/WDC-5m/path.pkl", 'rb') as f:
        paths = pickle.load(f)
    
    file_names = []
    for p in paths[0:100]:
        file_names.append(os.path.basename(p))
    logging.info(f"Start loading tokens dict")
    td = []
    for path in file_names:
        np = ('tokens_dict_') + path.removesuffix('.csv') + ".pkl"
        with open(os.path.join(tokens_dir_path, np), 'rb') as f:
            td.append(pickle.load(f))
    logging.info(f"Finish loading tokens dict")
    with Pool() as pool:
        results = pool.map(count_tokens, td)
    aggregated_tokens_counter = sum(results, Counter())
    # aggregated_tokens_counter = sum((Counter(token_dict) for token_dict in td), Counter())
    logging.info(f"Finish aggregating tokens dict")
    tokens_dict = {k: v for k, v in aggregated_tokens_counter.items()}
    logging.info(f"Finish getting tokens dict")
    with open(os.path.join(output_path, 'tokens_dict_5m_test_100_s.pkl'), 'wb') as f:
        pickle.dump(tokens_dict, f)
    return tokens_dict

if __name__ == '__main__':
    get_dict('/home/fatemeh/EDS-BaseLines/Uni-Detect/Utils/tokens_dir', '/home/fatemeh/EDS-BaseLines/Uni-Detect/Utils')
