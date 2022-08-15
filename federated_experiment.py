import logging
import os
import time
from datetime import datetime

import numpy.random
import torch
import yaml

from src.server import CentralServer
from src.utils import get_duration

if __name__ == '__main__':
    # read configuration file
    with open('./config.yaml') as c:
        configs = list(yaml.load_all(c, Loader=yaml.FullLoader))
    experiment_config = configs[0]["experiment_config"]
    attack_config = configs[1]['attack_settings']
    training_config = configs[2]["training_settings"]
    data_config = configs[3]["data_settings"]
    log_config = configs[4]["LOG_settings"]

    # read persistent file to create separate log_files
    with open('./log/persistent.txt', 'r', encoding="utf-8") as f:
        read_data = int(f.read())
        file_number = str(read_data + 1)

    # write back current number + 1 for next run.
    with open('./log/persistent.txt', 'w', encoding="utf-8") as f:
        f.write(file_number)

    # create date string for unique file_naming
    date_string = log_config['log_path'] \
                  + str(datetime.now().month) + '_' \
                  + str(datetime.now().day) + '/'

    # make directory per day for log files
    if not os.path.exists(date_string):
        os.makedirs(date_string)

    # create target model dir if not exists
    if not os.path.exists(experiment_config['model_path']):
        os.makedirs(experiment_config['model_path'])
        
    # create target model dir if not exists
    if not os.path.exists(attack_config['attack_model_path']):
        os.makedirs(attack_config['attack_model_path'])
        

    # Create log filename
    log_filename = date_string + file_number
    if experiment_config['train_model'] > 0:

        if attack_config['active_attack'] > 0:
            log_filename += '_active_target_training_'
        else:
            log_filename += '_target_training'
            
    elif attack_config['passive_attack'] > 0:
        if attack_config['active_attack'] > 0:
            log_filename += '_active_attack_'
        else:
            log_filename += '_passive_attack_'


    # Set op logger
    logging.basicConfig(filename=log_filename + '.txt',
                        filemode='w',
                        level=getattr(logging, log_config['log_level']))

    # Setup device
    device = "cuda" if torch.cuda.is_available() and experiment_config['device'] == "cuda" else "cpu"

    # set seeds
    for seed in experiment_config['seed']:
        print(seed)
        start_time = time.time()
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        numpy.random.seed(seed)

        # Setup Main Server with given parameters
        main_server = CentralServer(experiment_config,
                                    attack_config,
                                    data_config,
                                    training_config
                                    )
        main_server.start_up()

        # Log start message.
        message = f"[ Experiment with SEED: {seed} started! ]"
        logging.info(msg=message)
        # Perform main experiment
        main_server.perform_experiment()

        message = f"[ Total runtime: ... {str(get_duration(start_time))} for seed: {seed} ]"
        logging.info(message)
