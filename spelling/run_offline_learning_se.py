import logging
import os
import pickle
import sys
import time
import warnings

import yaml
from offline_learning_se import se_offline_learning

# Create the config file
with open(sys.argv[1]) as config_file:
    config = yaml.load(config_file, Loader=yaml.SafeLoader)

# Create the log file
os.makedirs(config["log_path"], exist_ok=True)
logging.basicConfig(
    filename=config["log_path"] + "/app.log",
    filemode="w",
    format="%(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)

with open(config["train_pkl_path"], "rb") as f:
    train = pickle.load(f)
    logging.info("train pickle file loaded.")

with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=RuntimeWarning)
    t0 = time.time()
    se_offline_learning(
        train, config["file_type"], config["n_cells_limit"], config["output_path"]
    )
    t1 = time.time()
    logging.info(f"SE Time: {t1-t0}")
