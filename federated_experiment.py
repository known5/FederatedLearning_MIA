import datetime
import logging
import os
import time
import yaml
import torch
from src.utils import get_duration

from src.server import CentralServer

if __name__ == '__main__':
    start_time = time.time()
    print("Experiment started!")
    # read configuration file
    with open('./config.yaml') as c:
        configs = list(yaml.load_all(c, Loader=yaml.FullLoader))
    experiment_config = configs[0]["experiment_config"]
    data_config = configs[1]["data_settings"]
    training_config = configs[2]["training_settings"]
    model_config = configs[3]["model_settings"]
    log_config = configs[4]["LOG_settings"]

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
    print(f"Total runtime: ... {str(get_duration(start_time))}")
