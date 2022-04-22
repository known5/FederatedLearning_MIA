import copy
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from multiprocessing import pool, cpu_count
from collections import OrderedDict

from src.client import Client
from src.attacker import Attacker
from src.utils import load_dataset, load_model
import torch.nn as nn


class CentralServer(object):
    """
    Ja hier moet dus documentatie
    """

    def __init__(self, experiment_param, data_param, fl_param, training_param, model_param, mia_param):
        """ Ja hier moet dus documentatie """
        self.seed = experiment_param['seed']
        self.device = experiment_param['device']
        self.is_mul = experiment_param['is_multi_threaded']
        self.perform_attack = experiment_param['perform_attack']
        self.evaluate_global_model = experiment_param['evaluate_global_model']

        self.dataset_path = data_param['data_path']
        self.dataset_name = data_param['dataset_name']
        self.split_ratio = data_param['split_ratio']
        self.is_idd = data_param['iid']

        self.number_of_clients = fl_param['number_of_clients']
        self.number_of_training_rounds = fl_param['training_rounds']

        self.training_param = training_param
        self.loss_function = training_param['loss_function']
        self.batch_size = training_param['batch_size']
        if not hasattr(nn, self.loss_function):
            error_message = f"...Optimizer \"{self.loss_function}\" is not supported or cannot be found in Torch " \
                            f"Optimizers! "
            raise AttributeError(error_message)
        else:
            self.loss_function = nn.__dict__[self.loss_function]()

        self.model_param = model_param
        self.attacker_present = mia_param['attacker_present']
        self.results = {"loss": [], "accuracy": []}

        self.test_data = None
        self.dataloader = None
        self.model = None
        self.clients = None

    def start_up(self):
        """ Ja hier moet dus documentatie """
        split_datasets, self.test_data = self.divide_datasets(self.dataset_name,
                                                              self.dataset_path,
                                                              self.number_of_clients,
                                                              self.is_idd)
        self.dataloader = DataLoader(self.test_data,
                                     batch_size=self.batch_size,
                                     shuffle=False)

        self.model = load_model(self.model_param['name'],
                                self.model_param['is_local_model'],
                                self.model_param['is_pre_trained'])
        self.clients = self.generate_clients(self.training_param)
        self.distribute_data_among_clients(split_datasets, self.split_ratio)
        self.share_model_with_clients()

    def generate_clients(self, training_param):
        """ Ja hier moet dus documentatie """
        print("Generating clients..... ")
        attacker_is_generated = False
        clients = []
        for client_id, dataset in enumerate(range(self.number_of_clients)):
            if self.attacker_present and not attacker_is_generated:
                clients.append(Attacker(client_id, training_param, device=self.device))
                attacker_is_generated = True
                print("Attacker generated...")
            else:
                clients.append(Client(client_id, training_param, device=self.device))
        print(f"Created {str(len(clients))} clients!")
        return clients

    def divide_datasets(self, data_name, data_path, number_of_clients, is_iid):
        """ Ja hier moet dus documentatie """
        # load correct dataset from torchvision
        training_data, test_data = load_dataset(data_path, data_name, number_of_clients, is_iid)
        # randomly split training data so each client has its own data set.
        print(f"Splitting datasets into {str(self.number_of_clients)} parts...")
        training_data_length = len(training_data) // self.number_of_clients
        length_split = [training_data_length for x in range(self.number_of_clients)]
        training_data_split = torch.utils.data.random_split(training_data, length_split)

        return training_data_split, test_data

    @staticmethod
    def train_clients(client):
        client.train()

    def share_model_with_clients(self):
        """ Ja hier moet dus documentatie """
        print("Sending global model to clients...")
        for client in self.clients:
            client.model = copy.deepcopy(self.model)
        print("Model sharing completed...")

    def distribute_data_among_clients(self, split_datasets, split_ratio):
        """ Ja hier moet dus documentatie """
        print("Loading datasets to clients...")
        for k, client in enumerate(self.clients):
            client.load_data(split_datasets[k], split_ratio)
        print("Sharing datasets, completed...")

    def aggregate_model(self):
        """ Ja hier moet dus documentatie """
        print("Aggregating models....")
        averaged_weights = OrderedDict()
        for i, client_id in enumerate(self.number_of_clients):
            local_weights = self.clients[client_id].model.state_dict()
            for layer_id in self.model.dict().keys():
                if i == 0:
                    averaged_weights[layer_id] = local_weights[layer_id]
                else:
                    averaged_weights[layer_id] = local_weights[layer_id]
        self.model.load_state_dict(averaged_weights)

    def do_training(self):
        """ Ja hier moet dus documentatie """
        # Share global model with all clients
        self.share_model_with_clients()
        # Do client training
        if self.is_mul:
            with pool.ThreadPool(processes=cpu_count() - 1) as workers:
                workers.map(self.train_clients, self.clients)
        # Aggregate the model updates into a new global model
        self.aggregate_model()

    def test_global_model(self):
        """ Ja hier moet dus documentatie """
        self.model.eval()
        self.model.to(self.device)

        test_loss, correct = 0, 0
        with torch.no_grad():
            for data, labels in self.dataloader:
                data, labels = data.float().to(self.device), labels.long().to(self.device)
                outputs = self.model(data)
                test_loss += self.loss_function(outputs, labels)

                predicted = outputs.argmax(dim=1, keepdim=True)
                correct += predicted.eq(labels.view_as(predicted)).sum().item()
        self.model.to("cpu")

        return test_loss / len(self.dataloader), correct / len(self.test_data)

    def perform_experiment(self, perform_attack=False, evaluate_global_model=False):
        """ Ja hier moet dus documentatie """
        round_counter = 0
        for index in tqdm(range(self.number_of_training_rounds)):

            client = self.clients[1]
            for data, labels in client.training_dataloader:
                print(data[0])
                break

            # round_counter += index + 1
            # self.do_training()
            # # If checked, perform global model evaluation every round.
            # if evaluate_global_model:
            #     round_loss, round_accuracy = self.test_global_model()
            #
            #     self.results['loss'].append(round_loss)
            #     self.results['accuracy'].append(round_accuracy)
            #
            #     print(f"[Round: {str(round_counter).zfill(4)}] Evaluate global model's performance...!\
            #         \n\t[Server] ...finished evaluation!\
            #         \n\t=> Loss: {round_loss:.4f}\
            #         \n\t=> Accuracy: {100. * round_accuracy:.2f}%\n")
            # # If checked, perform MIA during each round.
            # if perform_attack:
            #     pass
