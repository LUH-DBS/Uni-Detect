import os 
import pickle

def create_train_path_pkl(train_root_path, file_type, dest_path):
    train_path = []
    for root, dirs, files in os.walk(train_root_path):
        for file in files:
            if file.endswith(".{}".format(file_type)):
                train_path.append(os.path.join(root, file))
    with open(os.path.join(dest_path, 'test_path.pkl'), 'wb') as f:
        pickle.dump(train_path, f)

if __name__ == "__main__":
    train_root_path = "datasets/test"
    file_type = "csv"
    dest_path = "./"
    create_train_path_pkl(train_root_path, file_type, dest_path)