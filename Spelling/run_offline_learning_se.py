import time
import os
import sys

from offline_learning_se import se_offline_learning
import warnings
import pickle
import logging
import yaml

# Config file 
with open(sys.argv[1]) as config_file:
    config = yaml.load(config_file, Loader=yaml.SafeLoader)

logging.basicConfig(filename=config['log_path'] + '_app.log', filemode='w',
                    format='%(name)s - %(levelname)s - %(message)s', level=logging.INFO)

with open(config['train_pkl_path'], 'rb') as f:
    train = pickle.load(f)
    logging.info("train pickle file loaded.")

with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=RuntimeWarning)
    t0 = time.time()
    se_offline_learning(train, config['file_type'], config['output_path'])
    t1 = time.time()
    logging.info(f"SE Time: {t1-t0}")
