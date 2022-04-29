import os
import time
import yaml
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor

from src.server import CentralServer

# export LD_PRELOAD=tcmalloc.so:$LD_PRELOAD

if __name__ == '__main__':
    start_time = time.time()
    print("Experiment started!")
    # read configuration file
    with open('./config.yaml') as c:
        configs = list(yaml.load_all(c, Loader=yaml.FullLoader))
    experiment_config = configs[0]["experiment_config"]
    data_config = configs[1]["data_settings"]
    FL_config = configs[2]["FL_settings"]
    training_config = configs[3]["training_settings"]
    model_config = configs[4]["model_settings"]
    MIA_config = configs[5]["MIA_settings"]

    # Setup device
    device = "cuda" if torch.cuda.is_available() and experiment_config['device'] == "cuda" else "cpu"

    # Setup Main Server with given parameters
    main_server = CentralServer(experiment_config,
                                data_config,
                                FL_config,
                                training_config,
                                model_config,
                                MIA_config)
    main_server.start_up()
    main_server.perform_experiment(start_time, evaluate_global_model=experiment_config['evaluate_global_model'])

    hours, rem = divmod(time.time() - start_time, 3600)
    minutes, seconds = divmod(rem, 60)
    print("Total runtime: {:0>2}:{:0>2}:{:05.2f} ...".format(int(hours), int(minutes), seconds))
