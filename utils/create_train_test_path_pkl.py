import os
import pickle


def create_path_pkl(root_path: str, file_type: str, dest_path: str, pkl_file_name: str):
    """
    This function creates the path.pkl file which contains the path to all the files in the root_path.

    Parameters
    ----------
    root_path : str
        The path to the root directory.
    file_type : str
        The type of the files.
    dest_path : str
        The path to the destination directory.
    pkl_file_name : str
        The name of the pkl file.
    """
    path = []
    for root, dirs, files in os.walk(root_path):
        for file in files:
            if file.endswith(".{}".format(file_type)):
                path.append(os.path.join(root, file))
    os.makedirs(dest_path, exist_ok=True)
    with open(os.path.join(dest_path, pkl_file_name), "wb") as f:
        pickle.dump(path, f)


if __name__ == "__main__":
    test_root_path = "datasets/DGov-141/aggregated_dirty"
    file_type = "csv"
    dest_path = "output/DGov-141/"

    create_path_pkl(test_root_path, file_type, dest_path, "test_path.pkl")
