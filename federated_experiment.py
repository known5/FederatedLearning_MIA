import os
import time
import yaml
import torch
import logging
from src.utils import get_duration
from src.server import CentralServer
from datetime import datetime

if __name__ == '__main__':
    start_time = time.time()
    # read configuration file
    with open('./config.yaml') as c:
        configs = list(yaml.load_all(c, Loader=yaml.FullLoader))
    experiment_config = configs[0]["experiment_config"]
    data_config = configs[1]["data_settings"]
    training_config = configs[2]["training_settings"]
    model_config = configs[3]["model_settings"]
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

    # Set op logger
    logging.basicConfig(filename=date_string + file_number,
                        filemode='w',
                        level=getattr(logging, log_config['log_level']))

    message = "Experiment started!"
    logging.info(msg=message)

    # Setup device
    device = "cuda" if torch.cuda.is_available() and experiment_config['device'] == "cuda" else "cpu"

    # set seeds
    seed = experiment_config['seed']
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # Setup Main Server with given parameters
    main_server = CentralServer(experiment_config,
                                data_config,
                                training_config,
                                model_config
                                )
    main_server.start_up()
    # Perform main experiment
    main_server.perform_experiment(start_time)
    # Log scalars for tensorboard.
    message = f"[ Total runtime: ... {str(get_duration(start_time))} ]"
    logging.info(message)
