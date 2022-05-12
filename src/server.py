import copy
import time
import torch
from torch.utils.data import DataLoader
from collections import OrderedDict

from src.client import Client
from src.attacker import Attacker
from src.utils import load_dataset, load_model, get_duration
import torch.nn as nn


class CentralServer(object):
    """
    Ja hier moet dus documentatie
    """

    def __init__(self, experiment_param, data_param, training_param, model_param):
        """ Ja hier moet dus documentatie """
        self.device = experiment_param['device']
        self.do_local_eval = experiment_param['local_eval']
        self.do_global_eval = experiment_param['global_eval']
        self.do_passive_attack = experiment_param['passive_attack']

        self.dataset_path = data_param['data_path']
        self.dataset_name = data_param['dataset_name']
        self.split_ratio = data_param['local_split_ratio']
        self.is_idd = data_param['iid']

        self.training_param = training_param
        self.number_of_clients = training_param['number_of_clients']
        self.number_of_training_rounds = training_param['training_rounds']
        self.loss_function = training_param['loss_function']
        self.batch_size = training_param['batch_size']
        if not hasattr(nn, self.loss_function):
            error_message = f"...Optimizer \"{self.loss_function}\" is not supported or cannot be found in Torch " \
                            f"Optimizers! "
            raise AttributeError(error_message)
        else:
            self.loss_function = nn.__dict__[self.loss_function]()

        self.model_param = model_param
        self.results = {"loss": [], "accuracy": []}

        self.test_data = None
        self.dataloader = None
        self.model = load_model(self.model_param['name'],
                                self.model_param['is_local_model'])
        self.clients = None

    def start_up(self):
        """ Ja hier moet dus documentatie """
        split_datasets, self.test_data = self.divide_datasets(self.dataset_name,
                                                              self.dataset_path,
                                                              self.number_of_clients,
                                                              self.is_idd)
        self.dataloader = DataLoader(self.test_data,
                                     batch_size=self.batch_size,
                                     shuffle=False,
                                     num_workers=2,
                                     pin_memory=True
                                     )

        self.clients = self.generate_clients(self.training_param)
        self.distribute_data_among_clients(split_datasets, self.split_ratio)

    def generate_clients(self, training_param):
        """ Ja hier moet dus documentatie """
        attacker_is_generated = False
        clients = []
        for client_id, dataset in enumerate(range(self.number_of_clients)):
            if self.do_passive_attack > 0 and not attacker_is_generated:
                clients.append(Attacker(client_id, training_param, device=self.device))
                attacker_is_generated = True
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

    def share_model_with_clients(self):
        """ Ja hier moet dus documentatie """
        for client in self.clients:
            client.model = copy.deepcopy(self.model)

    def distribute_data_among_clients(self, split_datasets, split_ratio):
        """ Ja hier moet dus documentatie """
        for k, client in enumerate(self.clients):
            client.load_data(split_datasets[k], split_ratio)

    def aggregate_model(self):
        """ Ja hier moet dus documentatie """
        # print("Aggregating models....")
        averaged_dict = self.model.state_dict()
        for layer in averaged_dict.keys():
            averaged_dict[layer] = torch.stack([self.clients[i].model.state_dict()[layer].float() for i in range(len(self.clients))], 0).mean(0)
        self.model.load_state_dict(averaged_dict)

    def do_training(self, round_number):
        """ Ja hier moet dus documentatie """
        # Do client training
        for client in self.clients:
            client.train()

        if self.do_local_eval > 0:
            print(f"[Round: {str(round_number).zfill(4)}] Evaluate LOCAL model's performance...!")
            for client in self.clients:
                client.test()
                print(f"\n\t[Client {str(client.client_id)}] ...finished evaluation!\
                       \n\t=> Local Loss: {client.local_results.get('loss')[0]:.4f}\
                       \n\t=> Local Accuracy: {100. * client.local_results.get('accuracy')[0]:.2f}%")

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

    def perform_experiment(self, start_time=None):
        """ Ja hier moet dus documentatie """
        for index in range(self.number_of_training_rounds):
            round_time = time.time()
            hours, rem = divmod(round_time - start_time, 3600)
            minutes, seconds = divmod(rem, 60)
            current_time = "{:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds)
            print(f"[Round: {str(index).zfill(4)}] ... Started at {str(current_time)}")
            print("\t[Server]: ... share model!]")
            self.share_model_with_clients()
            print("\t[Server]: ... perform training!]")
            self.do_training(index)
            print("\t[Server]: ... aggregate model!]")
            self.aggregate_model()
            # If checked, perform global model evaluation every round.
        if self.do_global_eval > 0:
            round_loss, round_accuracy = self.test_global_model()

            self.results['loss'].append(round_loss)
            self.results['accuracy'].append(round_accuracy)

            print(f"[Round: {str(index).zfill(4)}] Evaluate global model's performance...!\
                \n\t[Server] ...finished evaluation!\
                \n\t=> Loss: {round_loss:.4f}\
                \n\t=> Accuracy: {100. * round_accuracy:.2f}%")

            # If checked, perform MIA during each round.
        if self.do_passive_attack():
            attacker = self.clients[0]

        print(f"[Round {str(index).zfill(4)} End] ... Round time: {str(get_duration(start_time))} s/it")
