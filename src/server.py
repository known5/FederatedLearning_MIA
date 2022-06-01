import copy
import logging
import shutil
import time
import torch
from torch.utils.data import DataLoader
import os

from src.client import Client
from src.attacker import Attacker
from src.utils import load_dataset, load_model, AverageMeter
import torch.nn as nn


# utils


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
        self.save_model = experiment_param['save_model']
        self.model_path = experiment_param['model_path']
        self.save_attack_model = experiment_param['save_attack_model']

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
            logging.error(error_message)
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
        message = 'Completed main server startup'
        logging.debug(message)

    def generate_clients(self, training_param):
        """ Ja hier moet dus documentatie """
        attacker_is_generated = False
        clients = []
        for client_id, dataset in enumerate(range(self.number_of_clients)):
            if self.do_passive_attack > 0 and not attacker_is_generated:
                clients.append(Attacker(client_id + 1,
                                        training_param,
                                        device=self.device,
                                        target_train_model=copy.deepcopy(self.model),
                                        exploit_layers=[24],
                                        exploit_gradients=[6]
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

    def divide_datasets(self, data_name, data_path, number_of_clients, is_iid):
        """ Ja hier moet dus documentatie """
        # load correct dataset from torchvision
        training_data, test_data, val_data = load_dataset(data_path, data_name, number_of_clients, is_iid)
        # randomly split training data so each client has its own data set.
        training_data_length = len(training_data) // self.number_of_clients
        message = f"Splitting dataset of size {len(training_data)}" \
                  f" into {self.number_of_clients}" \
                  f" parts of size {training_data_length}..."
        logging.info(message)
        length_split = [training_data_length for x in range(self.number_of_clients)]
        training_data_split = torch.utils.data.random_split(training_data, length_split)

        return training_data_split, test_data

    def share_model_with_clients(self):
        """ Ja hier moet dus documentatie """
        for client in self.clients:
            client.model = copy.deepcopy(self.model)
        message = 'Shared model with clients...'
        logging.debug(message)

    def distribute_data_among_clients(self, split_datasets, split_ratio):
        """ Ja hier moet dus documentatie """
        for k, client in enumerate(self.clients):
            client.load_data(split_datasets[k], split_ratio)
        message = 'Distributed data among clients'
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
        message = f'[ Round: {round_number} | Started! ]'
        logging.info(message)
        for client in self.clients:
            client.train(round_number)
            if self.do_local_eval > 0 and round_number % self.do_local_eval == 0:
                client.test(round_number)

        message = f'[ Round: {round_number} | Finished! ]'
        logging.info(message)

    def test_global_model(self, round_number):
        """ Ja hier moet dus documentatie """
        self.model.eval()
        self.model.to(self.device)

        batch_time = AverageMeter()
        losses = AverageMeter()
        accuracy = AverageMeter()

        dataset_size = len(self.dataloader)
        correct = 0

        start_time = time.time()
        message = f'[ Round: {round_number} | Global Model Eval started ]'
        logging.info(message)

        with torch.no_grad():
            for batch_idx, (data, labels) in enumerate(self.dataloader):
                data, labels = data.float().to(self.device), labels.long().to(self.device)
                outputs = self.model(data)
                loss = self.loss_function(outputs, labels)

                predicted = outputs.argmax(dim=1, keepdim=True)
                correct = predicted.eq(labels.view_as(predicted)).sum().item()

                # Update loss, accuracy and run_time metrics
                losses.update(loss.item(), self.batch_size)
                accuracy.update(correct, self.batch_size)
                batch_time.update(time.time() - start_time)

            # Create and log message about training
            message = f'[ Round: {round_number} ' \
                      f'| Time: {batch_time.avg:.2f}s ' \
                      f'| Loss: {losses.avg:.5f}' \
                      f'| Accuracy: {accuracy.avg:.2f}% ]'
            logging.info(message)
        self.model.to("cpu")

        return losses.avg, accuracy.avg

    def perform_experiment(self, start_time=None):
        """ Ja hier moet dus documentatie """
        round_loss, round_accuracy = 0, 0
        for index in range(self.number_of_training_rounds):
            # Do training cycle
            index += 1
            self.share_model_with_clients()
            self.do_training(index)
            self.aggregate_model()
            # If checked, perform global model evaluation every round.
            if self.do_global_eval > 0 and index % self.do_global_eval == 0:
                round_loss, round_accuracy = self.test_global_model(index)

                self.results['loss'].append(round_loss)
                self.results['accuracy'].append(round_accuracy)

            # If checked, perform MIA during each round.
            if self.do_passive_attack and index % self.do_passive_attack == 0:
                attacker = self.clients[0]
            if self.save_model > 1 and index % self.save_model == 0:
                is_best = round_accuracy > max(self.results['accuracy'])
                if is_best:
                    save_checkpoint({
                        'epoch': index,
                        'state_dict': self.model.state_dict(),
                        'acc': round_accuracy,
                        'best_acc': is_best
                    }, is_best=is_best,
                        filename=f'epoch_{index}_main',
                        checkpoint=self.model_path
                    )
            if self.save_attack_model > 1 and index % self.save_attack_model == 0:
                # TODO
                is_best = round_accuracy > max(self.results['accuracy'])
                if is_best:
                    save_checkpoint_adversary({
                        'epoch': index,
                        'state_dict': self.clients[0].model.state_dict(),
                        'acc': round_accuracy,
                        'best_acc': is_best
                    }, is_best=is_best,
                        filename=f'epoch_{index}_attack',
                        checkpoint=self.model_path
                    )
