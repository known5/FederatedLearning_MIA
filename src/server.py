import copy
import logging
import os
import shutil
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.attacker import Attacker
from src.client import Client
from src.models import AlexNet
from src.utils import load_dataset, AverageMeter, get_torch_loss_function


def save_checkpoint(state, is_best, checkpoint='checkpoint', filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth.tar'))


def save_checkpoint_adversary(state, is_best, checkpoint='checkpoint', filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_adversary_best.pth.tar'))


def load_target_model(path, filename):
    """ Ja hier moet dus documentatie """
    model = AlexNet()
    filepath = os.path.join(path, filename)
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['state_dict'])
    return model


class CentralServer(object):
    """
    Ja hier moet dus documentatie
    """

    def __init__(self, experiment_param, attack_param, data_param, training_param):
        """ Ja hier moet dus documentatie """
        self.device = experiment_param['device']
        self.train_model = experiment_param['train_model']
        self.do_local_eval = experiment_param['local_eval']
        self.do_global_eval = experiment_param['global_eval']
        self.save_model = experiment_param['save_model']
        self.model_path = experiment_param['model_path']

        self.attack_param = attack_param
        self.do_passive_attack = attack_param['passive_attack']
        self.do_active_attack = attack_param['active_attack']
        self.save_attack_model = attack_param['save_attack_model']
        self.attack_data_overlap = attack_param['attack_data_target_overlap_with']
        self.observed_target_models = attack_param['observed_target_models']
        self.attack_model_path = attack_param['attack_model_path']
        self.eval_attack = attack_param['eval_attack']

        self.dataset_path = data_param['data_path']
        self.dataset_name = data_param['dataset_name']
        self.number_of_classes = data_param['number_of_classes']

        self.training_param = training_param
        self.number_of_clients = training_param['number_of_clients']
        self.number_of_training_rounds = training_param['training_rounds']
        self.loss_function = get_torch_loss_function(training_param['loss_function'])
        self.batch_size = training_param['batch_size']
        self.client_data_overlap = training_param['client_data_overlap']
        self.if_overlap_client_dataset_size = training_param['if_overlap_client_dataset_size']

        self.results = {"loss": [], "accuracy": [0.0]}
        self.attack_results = {"loss": [], "accuracy": [0.0]}

        self.test_data = None
        self.global_test_dataloader = None
        self.clients = None
        self.model = None
        self.target_models_for_inference = []

    def start_up(self):
        """ Ja hier moet dus documentatie """

        self.model = AlexNet(self.number_of_classes)

        self.clients = self.generate_clients(self.training_param, self.attack_param)
        self.load_datasets(self.dataset_name,
                           self.dataset_path,
                           self.number_of_clients
                           )

        self.share_model_with_clients()

        if self.do_passive_attack > 0:
            for index in self.observed_target_models:
                filename = f'epoch_{index}_main_clients_{self.number_of_clients}'
                self.target_models_for_inference.append(load_target_model(self.model_path, filename))

        message = 'Completed main server startup'
        logging.debug(message)

    def generate_clients(self, training_param, attack_param):
        """ Ja hier moet dus documentatie """
        attacker_is_generated = False
        clients = []
        for client_id, dataset in enumerate(range(self.number_of_clients)):
            if self.do_passive_attack > 0 and not attacker_is_generated:
                clients.append(Attacker(client_id + 1,
                                        training_param,
                                        attack_param,
                                        device=self.device,
                                        target_train_model=copy.deepcopy(self.model),
                                        number_of_observed_models=len(self.observed_target_models)
                                        )
                               )
                attacker_is_generated = True
            else:
                clients.append(Client(client_id + 1,
                                      training_param,
                                      device=self.device,
                                      model=copy.deepcopy(self.model)))

        message = f"Created {len(clients)} clients!"
        logging.debug(message)

        return clients

    def load_datasets(self, data_name, data_path, number_of_clients):
        """ Ja hier moet dus documentatie """
        # load correct dataset from torchvision
        training_data, test_data = load_dataset(data_path, data_name)
        self.global_test_dataloader = DataLoader(test_data,
                                                 batch_size=self.batch_size,
                                                 shuffle=False,
                                                 num_workers=2,
                                                 pin_memory=True
                                                 )

        if self.client_data_overlap > 0:
            # randomly split training data so each client has its own separate data set.
            subset_length = len(training_data) // number_of_clients
            training_data_split = [subset_length for _ in range(self.number_of_clients)]
            remainder = len(training_data) % training_data_split[0]
            if not remainder == 0:
                training_data_split.append(remainder)
            training_data_split = torch.utils.data.random_split(training_data, training_data_split)

            message = f"Splitting dataset of size {len(training_data)}" \
                      f" into {self.number_of_clients}" \
                      f" parts of size {len(training_data_split[0])}..."
            logging.info(message)

            # send data to clients for training.
            for k, client in enumerate(self.clients):
                client.load_data(training_data_split[k])
            message = 'Distributed data among clients'
            logging.debug(message)

        else:
            # Give each client a fixed amount of samples, randomly drawn from the entire dataset. overlap may occur
            training_data_split = []
            temp_indices = np.arange(len(training_data))
            np.random.shuffle(temp_indices)

            for client in self.clients:
                temp_subset = []
                for i in range(self.if_overlap_client_dataset_size):
                    temp_subset.append(training_data[temp_indices[i]])
                client.load_data(temp_subset)
                training_data_split.append(temp_subset)
                np.random.shuffle(temp_indices)

            message = f"Splitting dataset of size {len(training_data)}" \
                      f" into {self.number_of_clients}" \
                      f" parts of size {len(training_data_split[0])}..."
            logging.info(message)

            message = 'Distributed data among clients'
            logging.debug(message)

        if self.do_passive_attack > 0:
            # If the condition below is met, sample member data from all clients
            if isinstance(self.attack_data_overlap, str) and self.attack_data_overlap == 'all':
                # Send data files to attacker for inference
                self.clients[0].load_attack_data(training_data, test_data)
            else:
                # Only provide member samples for 1 or more clients.
                attack_data = training_data_split[0]
                for index in range(1, self.attack_data_overlap):
                    attack_data = torch.utils.data.ConcatDataset([attack_data, training_data_split[index]])
                    # Send data files to attacker for inference
                self.clients[0].load_attack_data(attack_data, test_data)

    def share_model_with_clients(self):
        """ Ja hier moet dus documentatie """
        for client in self.clients:
            client.model = copy.deepcopy(self.model)
        message = 'Shared model with clients...'
        logging.debug(message)

    def aggregate_model(self):
        """ Ja hier moet dus documentatie """
        # print("Aggregating models....")
        averaged_dict = self.model.state_dict()
        for layer in averaged_dict.keys():
            averaged_dict[layer] = torch.stack(
                [self.clients[i].model.state_dict()[layer].float() for i in range(len(self.clients))], 0).mean(0)
        self.model.load_state_dict(averaged_dict)
        message = 'Aggregated model'
        logging.debug(message)

    def do_training(self, round_number):
        """ Ja hier moet dus documentatie """
        # Do client training
        for client in self.clients:
            client.train(round_number)
            if self.do_local_eval > 0 and round_number % self.do_local_eval == 0:
                client.test(round_number)

    def test_global_model(self, round_number):
        """ Ja hier moet dus documentatie """
        self.model.eval()
        self.model.to(self.device)

        batch_time = AverageMeter()
        losses = AverageMeter()

        data_size = len(self.global_test_dataloader.dataset)
        correct = 0

        message = f'[ Round: {round_number} | Global Model Eval started ]'
        logging.info(message)

        with torch.no_grad():
            start_time = time.time()
            for data, labels in self.global_test_dataloader:
                # Transfer data to device
                data, labels = data.float().to(self.device), labels.long().to(self.device)

                # Do a forward pass through the network to get prediction values and update loss metric.
                outputs = self.model(data)
                losses.update(self.loss_function(outputs, labels))

                # Compare predictions to labels and get accuracy score.
                _, predicted = torch.max(outputs.data, 1)
                correct += (predicted == labels).sum().item()

                # Update loss, accuracy and run_time metrics
                batch_time.update(time.time() - start_time)

            # Create and log message about test
            accuracy = (correct / data_size) * 100
            message = f'[ Round: {round_number} ' \
                      f'| Time: {batch_time.avg:.2f}s ' \
                      f'| Loss: {losses.avg:.5f}' \
                      f'| Accuracy: ({correct}/{data_size})={accuracy:.2f}% ]'
            logging.info(message)
        self.model.to("cpu")

        return losses.avg, accuracy

    def perform_experiment(self):
        """ Ja hier moet dus documentatie """
        round_loss, round_accuracy = 0, 0
        attacker = self.clients[0]
        for index in range(self.number_of_training_rounds):
            index += 1
            message = f'[ Round: {index} | Started! ]'

            # Adjust the learning rates at given epochs
            if index in [50, 100]:
                for client in self.clients:
                    client.learning_rate *= 0.1

            logging.info(message)
            # If checked, do training cycle
            if self.train_model > 0 and index % self.train_model == 0:
                self.do_training(index)
                self.aggregate_model()
                self.share_model_with_clients()
                # If checked, perform global model evaluation every round.
                if self.do_global_eval > 0 and index % self.do_global_eval == 0:
                    round_loss, round_accuracy = self.test_global_model(index)
                    self.results['loss'].append(round_loss)
                    self.results['accuracy'].append(round_accuracy)
                # If checked, save the current model and optimizer state
                if self.save_model > 0 and index % self.save_model == 0:
                    is_best = (round_accuracy >= max(self.results['accuracy']))
                    save_checkpoint({
                        'epoch': index,
                        'state_dict': self.model.state_dict(),
                        'acc': round_accuracy,
                        'best_acc': is_best
                    }, is_best=is_best,
                        filename=f'epoch_{index}'
                                 f'_main_clients_{self.number_of_clients}'
                                 f'_batch_{self.batch_size}',
                        checkpoint=self.model_path
                    )

            # If checked, perform passive MIA during each round.
            if self.do_passive_attack > 0 and index % self.do_passive_attack == 0:
                attacker.train_attack(index, self.target_models_for_inference)

                if self.eval_attack > 0 and index % self.eval_attack == 0:
                    attacker.test_attack(index, self.target_models_for_inference)

                if self.save_attack_model > 0 and index % self.save_attack_model == 0:
                    is_best = (round_accuracy >= max(self.attack_results['accuracy']))
                    save_checkpoint_adversary({
                        'epoch': index,
                        'state_dict': self.clients[0].model.state_dict(),
                        'acc': round_accuracy,
                        'best_acc': is_best,
                        'optimizer': attacker.attack_optimizer.state.dict()
                    }, is_best=is_best,
                        filename=f'_epoch_{index}'
                                 f'_attack_clients_{self.number_of_clients}'
                                 f'_batch_{attacker.attack_batch_size}',
                        checkpoint=self.attack_model_path
                    )

            message = f'[ Round: {index} | Finished! ]'
            logging.info(message)
