import copy
import logging
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.attacker import Attacker
from src.client import Client
from src.models import AlexNet
from src.utils import load_dataset, \
    AverageMeter, \
    get_torch_loss_function, \
    load_target_model, \
    save_checkpoint, \
    save_checkpoint_adversary


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
        self.adjust_lr_at_ = training_param['adjust_lr_at_']
        self.train_batch_size = training_param['train_batch_size']
        self.test_batch_size = training_param['test_batch_size']
        self.client_data_overlap = training_param['client_data_overlap']
        self.is_dataset_of_fixed_size = training_param['fixed_dataset']
        self.dataset_size = training_param['dataset_size']

        self.results = {"loss": [], "accuracy": [0.0]}
        self.attack_results = {"loss": [], "accuracy": [0.0]}

        self.test_data = None
        self.global_test_dataloader = None
        self.clients = None
        self.model = None
        self.target_models_for_inference = []

        message = f"[ Main Settings: \n" \
                  f"| Device: {self.device} \n" \
                  f"| Train Model: {self.train_model} \n" \
                  f"| global_eval: {self.do_global_eval} \n" \
                  f"| save_model: {self.save_model} \n" \
                  f"| clients: {self.number_of_clients} \n" \
                  f"| classes: {self.number_of_classes} \n" \
                  f"| learning_rate_schedule: {self.adjust_lr_at_} \n" \
                  f"| train_batch_size: {self.train_batch_size} \n" \
                  f"| test_batch_siez: {self.test_batch_size} \n" \
                  f"| rounds: {self.number_of_training_rounds} \n" \
                  f"| overlap: {self.client_data_overlap} \n" \
                  f"| fixed_dataset: {self.is_dataset_of_fixed_size} \n" \
                  f"| fixed_dataset_size: {self.dataset_size} \n" \
                  f"| passive_attack: {self.do_passive_attack} \n" \
                  f"| active_attack: {self.do_active_attack} \n" \
                  f"| eval_attack: {self.eval_attack} \n" \
                  f"| save_attack_model: {self.save_attack_model} \n" \
                  f"| observed_target_models: {self.observed_target_models} \n" \
                  f"| model_path: {self.model_path} \n" \
                  f"| Attack_model_path: {self.attack_model_path} ]\n"
        logging.info(msg=message)

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
                filename = f'epoch_{index}_main_clients_{self.number_of_clients}_batch_{self.train_batch_size}'
                self.target_models_for_inference.append(load_target_model(self.model_path, filename))

        message = 'Completed main server startup'
        logging.debug(message)

    def generate_clients(self, training_param, attack_param):
        """ Ja hier moet dus documentatie """
        attacker_is_generated = False
        clients = []
        for client_id, dataset in enumerate(range(self.number_of_clients)):
            if (self.do_passive_attack > 0 or self.do_active_attack != []) and not attacker_is_generated:
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
        train_data, test_data = load_dataset(data_path, data_name)
        self.global_test_dataloader = DataLoader(test_data,
                                                 batch_size=self.test_batch_size,
                                                 shuffle=False,
                                                 num_workers=2,
                                                 pin_memory=True
                                                 )

        # gebruik list om aantal per class te weten
        if self.is_dataset_of_fixed_size == 1:
            data_size = self.dataset_size
        else:
            data_size = len(train_data) // self.number_of_clients

        client_choice_lists = []
        if self.client_data_overlap == 1:
            train_set = range(len(train_data))
            for client in range(number_of_clients):
                choices = np.random.choice(train_set, size=data_size, replace=True)
                client_choice_lists.append(list(choices))
                client_subset = torch.utils.data.Subset(train_data, choices)
                self.clients[client].load_data(client_subset)

        else:
            client_indices = []
            for _ in range(self.number_of_clients):
                client_indices.append([])
            for data_class in range(self.number_of_classes):
                # Set the boundaries for picking samples from the right classes
                low, high = data_class * 1000, (data_class + 1) * 1000
                random_subset = list(range(low, high))
                random.shuffle(random_subset)
                length = self.dataset_size // self.number_of_classes
                for _ in range(length):
                    for client_list in client_indices:
                        client_list.append(random_subset.pop())
            for client in range(number_of_clients):
                choices = client_indices[client]
                client_choice_lists.append(list(choices))
                client_subset = torch.utils.data.Subset(train_data, choices)
                self.clients[client].load_data(client_subset)

        if self.do_passive_attack > 0 or self.do_active_attack != []:
            # If the condition below is met, sample member data from all clients
            if isinstance(self.attack_data_overlap, str) and self.attack_data_overlap == 'all':
                # Send data files to attacker for inference
                self.clients[0].load_attack_data(train_data, test_data)
            else:
                # Only provide member samples for 1 or more clients.
                attack_data_indices = []
                for index in range(self.attack_data_overlap):
                    attack_data_indices += client_choice_lists[index]
                    # Send data files to attacker for inference
                self.clients[0].load_attack_data(train_data, test_data, attack_data_indices, data_size)

        message = f"Dataset loaded and divided into {data_size} samples per client"
        if self.client_data_overlap == 1:
            message += ' with overlap '
        else:
            message += ' with no overlap '
        logging.info(message)

        message = 'Distributed data among clients'
        logging.debug(message)

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

            logging.info(message)
            # If checked, do training cycle
            if self.train_model > 0 and index % self.train_model == 0:
                # Adjust the learning rates at given epochs
                if index in self.adjust_lr_at_:
                    for client in self.clients:
                        client.learning_rate *= 0.1

                self.do_training(index)
                # If checked, perform gradient ascent learning on the attacker dataset each round.
                if index in self.do_active_attack and self.do_active_attack != []:
                    attacker.gradient_ascent_attack(index)

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
                                 f'_batch_{self.train_batch_size}',
                        checkpoint=self.model_path
                    )

            # If checked, perform passive MIA during each round.
            if self.do_passive_attack > 0 and index % self.do_passive_attack == 0:
                attacker.train_attack(index, self.target_models_for_inference)

                if self.eval_attack > 0 and index % self.eval_attack == 0:
                    round_loss, round_accuracy = attacker.test_attack(index, self.target_models_for_inference)
                    self.attack_results['loss'].append(round_loss)
                    self.attack_results['accuracy'].append(round_accuracy)

                if self.save_attack_model > 0 and index % self.save_attack_model == 0:
                    is_best = (round_accuracy >= max(self.attack_results['accuracy']))
                    if is_best:
                        save_checkpoint_adversary({
                            'epoch': index,
                            'state_dict': self.clients[0].model.state_dict(),
                            'acc': round_accuracy,
                            'best_acc': is_best,
                            'optimizer': attacker.attack_optimizer.state_dict()
                        }, is_best=is_best,
                            filename=f'_attack_clients_{self.number_of_clients}'
                                     f'_batch_{attacker.attack_test_batch_size}',
                            checkpoint=self.attack_model_path
                        )

            message = f'[ Round: {index} | Finished! ]'
            logging.info(message)
