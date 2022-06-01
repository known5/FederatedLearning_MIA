import logging

import torch.utils
import torch.optim as optimizers
import torch.nn as nn
from torch.utils.data import DataLoader
from src.utils import *


class Client(object):

    def __init__(self, client_id, training_param, device, model):
        """ Ja hier moet dus documentatie """
        self.__model = None
        self.loss_function = training_param['loss_function']
        if not hasattr(nn, self.loss_function):
            error_message = f"...Loss Function: \"{self.loss_function}\" is not supported or cannot be found in " \
                            f"Torch Optimizers! "
            logging.error(error_message)
            raise AttributeError(error_message)
        else:
            self.loss_function = nn.__dict__[self.loss_function]()
        self.number_of_epochs = training_param['epochs']
        self.model = model
        self.optimizer_name = training_param['optimizer']
        self.learning_rate = training_param['learning_rate']
        self.momentum = training_param['momentum']

        self.batch_size = training_param['batch_size']
        self.client_id = client_id
        self.device = device
        self.data = None

        self.training_dataloader = None
        self.testing_dataloader = None
        self.optimizer = None
        self.local_results = {"loss": [], "accuracy": []}

    @property
    def model(self):
        """ Ja hier moet dus documentatie """
        return self.__model

    @model.setter
    def model(self, model):
        """ Ja hier moet dus documentatie """
        self.__model = model

    def load_data(self, training_data, split_ratio):
        """ Ja hier moet dus documentatie """
        train_set_size = int(len(training_data) * split_ratio)
        test_set_size = len(training_data) - train_set_size
        train_set, self.data = torch.utils.data.random_split(training_data, [train_set_size, test_set_size])
        self.training_dataloader = DataLoader(train_set,
                                              batch_size=self.batch_size,
                                              shuffle=True,
                                              num_workers=2,
                                              pin_memory=False
                                              )
        self.testing_dataloader = DataLoader(self.data,
                                             batch_size=self.batch_size,
                                             shuffle=True,
                                             num_workers=2,
                                             pin_memory=False
                                             )
        message = f'Client {self.client_id} loaded datasets successfully'
        logging.debug(message)

    def train(self, round_number):
        """ Ja hier moet dus documentatie """
        self.model.train()
        self.model.to(self.device)

        dataset_size = len(self.training_dataloader.dataset)

        self.optimizer = optimizers.__dict__[self.optimizer_name](
            params=self.model.parameters(),
            lr=self.learning_rate,
            momentum=self.momentum
        )

        for e in range(self.number_of_epochs):
            batch_time = AverageMeter()
            losses = AverageMeter()
            accuracy = AverageMeter()
            start_time = time.time()
            correct = 0
            for data, labels in self.training_dataloader:
                # Transfer data to CPU or GPU and set gradients to zero for performance.
                data, labels = data.float().to(self.device), labels.long().to(self.device)
                self.optimizer.zero_grad()

                # Do a forward pass through the network to get prediction values and update loss metric.
                outputs = self.model(data)
                loss = self.loss_function(outputs, labels)

                # Do a backward pass through the network to get the gradients
                # and then use the optimizer to update the weights.
                loss.backward()
                self.optimizer.step()

                # Compare predictions to labels and get accuracy score.
                predicted = outputs.argmax(dim=1, keepdim=True)
                correct += predicted.eq(labels.view_as(predicted)).sum().item()

                # Update loss, accuracy and run_time metrics
                losses.update(loss.item())
                accuracy.update((correct / dataset_size) * 100)
                batch_time.update(time.time() - start_time)

            message = f'[ Round: {round_number} ' \
                      f'| Local Train ' \
                      f'| Client: {self.client_id} ' \
                      f'| Epoch: {e + 1} ' \
                      f'| Time: {batch_time.avg:.2f}s ' \
                      f'| Loss: {losses.avg:.5f} ' \
                      f'| Train Accuracy {accuracy.avg:.2f}% ]'
            logging.info(message)
        self.model.to("cpu")

    def test(self, round_number):
        """ Ja hier moet dus documentatie """
        self.model.eval()
        self.model.to(self.device)

        dataset_size = len(self.training_dataloader.dataset)

        losses = AverageMeter()
        accuracy = AverageMeter()
        batch_time = AverageMeter()
        correct = 0
        with torch.no_grad():
            start_time = time.time()
            for data, labels in self.training_dataloader:
                # Transfer data to CPU or GPU.
                data, labels = data.float().to(self.device), labels.long().to(self.device)

                # Do a forward pass through the network to get prediction values and update loss metric.
                outputs = self.model(data)
                losses.update(self.loss_function(outputs, labels))

                # Compare predictions to labels and get accuracy score.
                predicted = outputs.argmax(dim=1, keepdim=True)
                correct += predicted.eq(labels.view_as(predicted)).sum().item()

                # Update time and accuracy metric.
                accuracy.update((correct / dataset_size) * 100)
                batch_time.update(time.time() - start_time)

            message = f'[ Round: {round_number} ' \
                      f'| Local Eval ' \
                      f'| Time: {batch_time.avg:.2f}s ' \
                      f'| Client: {self.client_id} ' \
                      f'| Loss: {losses.avg:.5f} ' \
                      f'| Train Accuracy {accuracy.avg:.2f}% ]'
            logging.info(message)

        self.local_results = {"loss": [], "accuracy": []}
        self.local_results['loss'].append(losses.avg)
        self.local_results['accuracy'].append(accuracy.avg)
