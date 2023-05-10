import time
import warnings
import pickle
import logging
import sys
import yaml
import fd_offline_learning as fd

# Load config file
with open(sys.argv[1]) as config_file:
    config = yaml.load(config_file, Loader=yaml.SafeLoader)

# Set up logging
logging.basicConfig(filename=config['log_path'] + '_app.log', filemode='w',
                    format='%(name)s - %(levelname)s - %(message)s', level=logging.INFO)

with open(config['train_pkl_path'], 'rb') as f:
    train = pickle.load(f)
    logging.info("train pickle file loaded.")

with open(config['tokens_dict_path'], 'rb') as f:
    tokens_dict = pickle.load(f)
    logging.info("tokens_dict pickle file loaded.")

with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=RuntimeWarning)
    t0 = time.time()
    fd.fd_offline_learning(train, config['file_type'], config['output_path'], tokens_dict)
    t1 = time.time()
    logging.info(f"Time: {t1-t0}")
