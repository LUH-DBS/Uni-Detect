import os
import random
import shutil

def split_data(clean_folder_path: str, dirty_folder_path: str, train_folder_path: str, test_folder_path: str, train_percent: float) -> None:
    """
    Randomly splits clean and dirty CSV files from given folders into train and test folders.

    Parameters
    ----------
    clean_folder_path: path to folder containing clean CSV files
    dirty_folder_path: path to folder containing dirty CSV files
    train_folder_path: path to folder to store train CSV files
    test_folder_path: path to folder to store test CSV files
    train_percent: percentage of data to use for training (between 0 and 1)
    """
    if train_percent < 0 or train_percent > 1:
        raise ValueError("train_percent must be between 0 and 1")

    # Create train and test folders if they don't exist
    train_clean_folder_path = os.path.join(train_folder_path, "clean")
    train_dirty_folder_path = os.path.join(train_folder_path, "dirty")
    test_clean_folder_path = os.path.join(test_folder_path, "clean")
    test_dirty_folder_path = os.path.join(test_folder_path, "dirty")
    os.makedirs(train_clean_folder_path, exist_ok=True)
    os.makedirs(train_dirty_folder_path, exist_ok=True)
    os.makedirs(test_clean_folder_path, exist_ok=True)
    os.makedirs(test_dirty_folder_path, exist_ok=True)

    # Get list of file names in clean folder
    file_names = os.listdir(clean_folder_path)

    # Shuffle file names
    random.shuffle(file_names)

    # Calculate number of files to move to train folder
    num_train = int(train_percent * len(file_names))

    # Move files from clean folder to train and test folders
    for i, file_name in enumerate(file_names):
        # Determine source and destination paths
        clean_file_path = os.path.join(clean_folder_path, file_name)
        dirty_file_path = os.path.join(dirty_folder_path, file_name)
        if i < num_train:
            # Move to train folder
            dest = {"clean": os.path.join(train_clean_folder_path, file_name), "dirty": os.path.join(train_dirty_folder_path, file_name)}
        else:
            # Move to test folder
            dest = {"clean": os.path.join(test_clean_folder_path, file_name), "dirty": os.path.join(test_dirty_folder_path, file_name)}
        shutil.copyfile(clean_file_path, dest["clean"])
        shutil.copyfile(dirty_file_path, dest["dirty"])

if __name__ == "__main__":
    clean_folder_path = "/home/fatemeh/EDS-BaseLines/Uni-Detect/datasets/data-gov/aggregated_clean"
    dirty_folder_path = "/home/fatemeh/EDS-BaseLines/Uni-Detect/datasets/data-gov/aggregated_dirty"
    train_folder_path = "/home/fatemeh/EDS-BaseLines/Uni-Detect/datasets/data-gov/train"
    test_folder_path = "/home/fatemeh/EDS-BaseLines/Uni-Detect/datasets/data-gov/test"
    train_percent = 0.8
    split_data(clean_folder_path, dirty_folder_path, train_folder_path, test_folder_path, train_percent)