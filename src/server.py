import copy
import logging
import os
import shutil
import time

import torch
from torch.utils.data import DataLoader

from src.attacker import Attacker
from src.client import Client
from src.utils import load_dataset, load_model, AverageMeter, get_torch_loss_function


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

    def __init__(self, experiment_param, attack_param, data_param, training_param, model_param):
        """ Ja hier moet dus documentatie """
        self.device = experiment_param['device']
        self.train_model = experiment_param['train_model']
        self.do_local_eval = experiment_param['local_eval']
        self.do_global_eval = experiment_param['global_eval']
        self.save_model = experiment_param['save_model']
        self.model_path = experiment_param['model_path']

        self.attack_param = attack_param
        self.do_passive_attack = attack_param['passive_attack']
        self.save_attack_model = attack_param['save_attack_model']
        self.load_target_models = attack_param['load_target_models']

        self.dataset_path = data_param['data_path']
        self.dataset_name = data_param['dataset_name']
        self.is_idd = data_param['iid']

        self.training_param = training_param
        self.number_of_clients = training_param['number_of_clients']
        self.number_of_training_rounds = training_param['training_rounds']
        self.loss_function = get_torch_loss_function(training_param['loss_function'])
        self.batch_size = training_param['batch_size']

        self.model_param = model_param
        self.results = {"loss": [], "accuracy": [0.0]}
        self.attack_results = {"loss": [], "accuracy": [0.0]}

        self.test_data = None
        self.global_test_dataloader = None
        self.clients = None
        self.model = None

    def start_up(self):
        """ Ja hier moet dus documentatie """

        self.model = load_model(self.model_param['name'],
                                self.model_param['is_local_model'])
        self.clients = self.generate_clients(self.training_param, self.attack_param)
        self.load_datasets(self.dataset_name,
                           self.dataset_path,
                           self.number_of_clients,
                           self.is_idd)

        self.share_model_with_clients()
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
                                        target_path=self.model_path,
                                        load_models=self.load_target_models
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

    def load_datasets(self, data_name, data_path, number_of_clients, is_iid):
        """ Ja hier moet dus documentatie """
        # load correct dataset from torchvision
        training_data, test_data = load_dataset(data_path, data_name)
        self.global_test_dataloader = DataLoader(test_data,
                                                 batch_size=self.batch_size,
                                                 shuffle=False,
                                                 num_workers=2,
                                                 pin_memory=True
                                                 )

        # randomly split training data so each client has its own data set.
        training_data_length = len(training_data) // number_of_clients
        length_split = [training_data_length for _ in range(self.number_of_clients)]
        training_data_split = torch.utils.data.random_split(training_data, length_split)

        message = f"Splitting dataset of size {len(training_data)}" \
                  f" into {self.number_of_clients}" \
                  f" parts of size {training_data_length}..."
        logging.info(message)

        # send data to clients for training.
        for k, client in enumerate(self.clients):
            client.load_data(training_data_split[k])
        message = 'Distributed data among clients'
        logging.debug(message)

        if self.do_passive_attack > 0:
            # Send data files to attacker for inference
            self.clients[0].load_attack_data(training_data, test_data)

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
        accuracy = AverageMeter()

        correct = 0
        total = 0

        start_time = time.time()
        message = f'[ Round: {round_number} | Global Model Eval started ]'
        logging.info(message)

        with torch.no_grad():
            for batch_idx, (data, labels) in enumerate(self.global_test_dataloader):
                data, labels = data.float().to(self.device), labels.long().to(self.device)
                outputs = self.model(data)
                loss = self.loss_function(outputs, labels)

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                # Update loss, accuracy and run_time metrics
                losses.update(loss.item())
                accuracy.update(correct, self.batch_size)
                batch_time.update(time.time() - start_time)

            # Create and log message about training
            message = f'[ Round: {round_number} ' \
                      f'| Time: {batch_time.avg:.2f}s ' \
                      f'| Loss: {losses.avg:.5f}' \
                      f'| Accuracy: ({correct}/{total})={(100 * correct / total):.2f}% ]'
            logging.info(message)
        self.model.to("cpu")

        return losses.avg, accuracy.avg

    def perform_experiment(self):
        """ Ja hier moet dus documentatie """
        round_loss, round_accuracy = 0, 0
        attacker = self.clients[0]
        for index in range(self.number_of_training_rounds):
            index += 1
            message = f'[ Round: {index} | Started! ]'
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
                        filename=f'epoch_{index}_main_clients_{self.number_of_clients}',
                        checkpoint=self.model_path
                    )

            # If checked, perform MIA during each round.
            if self.do_passive_attack > 0 and index % self.do_passive_attack == 0:
                attacker.perform_attack()

                if self.save_attack_model > 0 and index % self.save_attack_model == 0:
                    is_best = (round_accuracy >= max(self.attack_results['accuracy']))
                    save_checkpoint_adversary({
                        'epoch': index,
                        'state_dict': self.clients[0].model.state_dict(),
                        'acc': round_accuracy,
                        'best_acc': is_best,
                        'optimizer': attacker.attack_optimizer.state.dict()
                    }, is_best=is_best,
                        filename=f'epoch_{index}_attack_clients_{self.number_of_clients}',
                        checkpoint=self.model_path
                    )

            message = f'[ Round: {index} | Finished! ]'
            logging.info(message)
