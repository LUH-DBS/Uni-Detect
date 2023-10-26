import multiprocessing
import os
import pickle

from tqdm import tqdm


def getlist_of_dicts(tokens_dir_path):
    with open(
        "/home/fatemeh/EDS-BaseLines/Uni-Detect/output/WDC-5m/path.pkl", "rb"
    ) as f:
        paths = pickle.load(f)

    file_names = [os.path.basename(p) for p in paths[0:1000000]]
    print("Start loading tokens dict")
    td = []
    for path in file_names:
        np = "tokens_dict_" + path.removesuffix(".csv") + ".pkl"
        with open(os.path.join(tokens_dir_path, np), "rb") as f:
            td.append(pickle.load(f))
            if len(td) % 1000000:
                print(len(td))
    print("Finish loading tokens dict")
    return td


def merge_dicts(dicts):
    result = {}
    for dictionary in dicts:
        for key, value in dictionary.items():
            result[key] = result.get(key, 0) + value
    return result


def merge_dicts_parallel(list_of_dicts, num_processes, output_path):
    pool = multiprocessing.Pool(num_processes)
    chunk_size = len(list_of_dicts) // num_processes
    chunks = [
        list_of_dicts[i : i + chunk_size]
        for i in range(0, len(list_of_dicts), chunk_size)
    ]

    merged_dict = {}
    with tqdm(total=len(chunks)) as pbar:
        for result in pool.imap_unordered(merge_dicts, chunks):
            for key, value in result.items():
                merged_dict[key] = merged_dict.get(key, 0) + value
            pbar.update(1)

    with open(os.path.join(output_path, "tokens_dict_1m_mrjob.pkl"), "wb") as f:
        pickle.dump(merged_dict, f)
    return merged_dict


if __name__ == "__main__":
    td = getlist_of_dicts("/home/fatemeh/EDS-BaseLines/Uni-Detect/utils/tokens_dir")
    merge_dicts_parallel(td, 64, "/home/fatemeh/EDS-BaseLines/Uni-Detect/output/WDC-1m")
