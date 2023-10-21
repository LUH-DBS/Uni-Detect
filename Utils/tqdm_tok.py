from multiprocessing import Pool
import os
import logging
import pickle
from collections import Counter
from tqdm import tqdm  # Import tqdm library for the progress bar

def count_tokens(token_dict):
    return Counter(token_dict)

def get_dict(tokens_dir_path, output_path):
    with open("/home/fatemeh/EDS-BaseLines/Uni-Detect/output/WDC-5m/path.pkl", 'rb') as f:
        paths = pickle.load(f)

    file_names = []
    for p in paths:
        file_names.append(os.path.basename(p))
    logging.info(f"Start loading tokens dict")
    td = []
    for path in tqdm(file_names, desc="Loading tokens dict", unit="file"):
        np = ('tokens_dict_') + path.removesuffix('.csv') + ".pkl"
        with open(os.path.join(tokens_dir_path, np), 'rb') as f:
            td.append(pickle.load(f))
            if len(td) % 10000 == 0:
                print(f"Finish loading {len(td)} tokens dict")
    print(f"Finish loading tokens dict")
    n_tables = len(td)
    with Pool() as pool:
        results = list(tqdm(pool.imap(count_tokens, td), total=len(td), desc="Processing tokens", unit="table"))
    aggregated_tokens_counter = tqdm(sum(results, Counter()), desc="Aggregating tokens", unit="token")
    print(f"Finish aggregating tokens dict")
    tokens_dict = {k: v / n_tables for k, v in tqdm(aggregated_tokens_counter.items(), desc="Creating tokens dict", unit="item")}
    print(f"Finish getting tokens dict")
    with open(os.path.join(output_path, 'tokens_dict_5m_tqdm.pkl'), 'wb') as f:
        pickle.dump(tokens_dict, f)
    return tokens_dict

if __name__ == '__main__':
    get_dict('/home/fatemeh/EDS-BaseLines/Uni-Detect/Utils/tokens_dir', '/home/fatemeh/EDS-BaseLines/Uni-Detect/Utils')
