import os
import pickle
import shutil


def create_aggregated_lakes(
    separated_sandbox_path,
    aggregated_dirty_sandbox_path,
    aggregated_clean_sandbox_path,
    dirty_file_name,
    clean_file_name,
    results_path,
):
    """
    This function creates the aggregated lake from the separated sandbox.
    It moves the dirty_clean.csv and clean.csv files from each sandbox to the aggregated lakes.
    It also creates a dictionary of the tables and their new names.

    Parameters
    ----------
    separated_sandbox_path : str
        The path to the separated sandbox.
    aggregated_dirty_sandbox_path : str
        The path to the aggregated dirty sandbox.
    aggregated_clean_sandbox_path : str
        The path to the aggregated clean sandbox.
    dirty_file_name : str
        The name of the dirty file.
    clean_file_name : str
        The name of the clean file.
    """
    count = 0
    tables_dict = {}
    for subdir, dirs, files in os.walk(separated_sandbox_path):
        if subdir != separated_sandbox_path:
            print(subdir)
            new_table_name = "{}.csv".format(count)
            tables_dict[os.path.basename(subdir)] = new_table_name
            # Move dirty_clean.csv to aggregated_dirty_sandbox_path
            file = dirty_file_name
            src = os.path.join(subdir, file)
            dst = os.path.join(aggregated_dirty_sandbox_path, new_table_name)
            shutil.copy(src, dst)

            # Move clean.csv to aggregated_clean_sandbox_path
            file = clean_file_name
            src = os.path.join(subdir, file)
            dst = os.path.join(aggregated_clean_sandbox_path, new_table_name)
            shutil.copy(src, dst)

            count += 1
    print("table_dict", tables_dict)
    with open(os.path.join(results_path, "tables_dict.pickle"), "wb") as handle:
        pickle.dump(tables_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)


separated_sandbox_path = (
    "/home/fatemeh/EDS-BaseLines/Uni-Detect/datasets/DGov-141/Separated"
)
aggregated_dirty_sandbox_path = (
    "/home/fatemeh/EDS-BaseLines/Uni-Detect/datasets/DGov-141/aggregated_dirty"
)
aggregated_clean_sandbox_path = (
    "/home/fatemeh/EDS-BaseLines/Uni-Detect/datasets/DGov-141/aggregated_clean"
)
if not os.path.exists(aggregated_dirty_sandbox_path):
    os.mkdir(aggregated_dirty_sandbox_path)
if not os.path.exists(aggregated_clean_sandbox_path):
    os.mkdir(aggregated_clean_sandbox_path)
results_path = "/home/fatemeh/EDS-BaseLines/Uni-Detect/datasets/DGov-141/"
dirty_file_name = "dirty.csv"
clean_file_name = "clean.csv"

create_aggregated_lakes(
    separated_sandbox_path,
    aggregated_dirty_sandbox_path,
    aggregated_clean_sandbox_path,
    dirty_file_name,
    clean_file_name,
    results_path,
)
