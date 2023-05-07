import time
import sys
from offline_learning_no import no_offline_learning
import warnings
import pickle
import yaml
import logging
import sys

# Create config file
with open(sys.argv[1]) as config_file:
    config = yaml.load(config_file, Loader=yaml.SafeLoader)

# Creating log files
logging.basicConfig(filename=config['log_path'] + '_app.log', filemode='w',
                    format='%(name)s - %(levelname)s - %(message)s', level=logging.INFO)

# Load train tables path pickle file
with open(config['train_pkl_path'], 'rb') as f:
    train_path_list = pickle.load(f)
    logging.info("train pickle file loaded.")

# Run offline learning
with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=RuntimeWarning)
    t0 = time.time()
    no_offline_learning(train_path_list, config['file_type'], config['output_path'])
    t1 = time.time()
    logging.info(f"No Time: {t1-t0}")
