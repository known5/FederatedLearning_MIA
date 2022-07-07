import logging
import time

import torch.optim as optimizers
import torch.utils
from torch.utils.data import DataLoader

from src.utils import AverageMeter, get_torch_loss_function


class Client(object):

    def __init__(self, client_id, training_param, device, model):
        """ Ja hier moet dus documentatie """
        self.__model = model
        self.loss_function = get_torch_loss_function(training_param['loss_function'])
        self.optimizer_name = training_param['optimizer']
        self.learning_rate = training_param['learning_rate']
        self.momentum = training_param['momentum']
        self.weight_decay = training_param['weight_decay']

        self.batch_size = training_param['batch_size']
        self.client_id = client_id
        self.device = device

        self.training_data = None
        self.test_data = None
        self.training_dataloader = None
        self.testing_dataloader = None

        self.local_results = {"loss": [], "accuracy": []}

    @property
    def model(self):
        """ Ja hier moet dus documentatie """
        return self.__model

    @model.setter
    def model(self, model):
        """ Ja hier moet dus documentatie """
        self.optimizer = optimizers.__dict__[self.optimizer_name](
            params=model.parameters(),
            lr=self.learning_rate,
            momentum=self.momentum,
            weight_decay=self.weight_decay
        )
        self.__model = model

    def load_data(self, training_data):
        """ Ja hier moet dus documentatie """
        self.training_data = training_data
        self.training_dataloader = DataLoader(self.training_data,
                                              batch_size=self.batch_size,
                                              shuffle=True,
                                              num_workers=2,
                                              pin_memory=True
                                              )

        message = f'Client {self.client_id} loaded datasets successfully'
        logging.debug(message)

    def train(self, round_number):
        """ Ja hier moet dus documentatie """
        self.model.train()
        self.model.to(self.device)

        data_size = len(self.training_dataloader.dataset)

        batch_time = AverageMeter()
        losses = AverageMeter()

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
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()

            # Update loss, accuracy and run_time metrics
            losses.update(loss.item())
            batch_time.update(time.time() - start_time)

        message = f'[ Round: {round_number} ' \
                  f'| Local Train ' \
                  f'| Client: {self.client_id} ' \
                  f'| Time: {batch_time.avg:.2f}s ' \
                  f'| Loss: {losses.avg:.5f} ' \
                  f'| Tr_Acc ({correct}/{data_size})={((correct / data_size) * 100):.2f}% ]'
        logging.info(message)
        self.model.to("cpu")

    def test(self, round_number):
        """ Ja hier moet dus documentatie """
        self.model.eval()
        self.model.to(self.device)

        data_size = len(self.training_dataloader.dataset)

        losses = AverageMeter()
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
                _, predicted = torch.max(outputs.data, 1)
                correct += (predicted == labels).sum().item()

                # Update time and accuracy metric.
                batch_time.update(time.time() - start_time)

            accuracy = (correct / data_size) * 100
            message = f'[ Round: {round_number} ' \
                      f'| Local Eval ' \
                      f'| Time: {batch_time.avg:.2f}s ' \
                      f'| Client: {self.client_id} ' \
                      f'| Loss: {losses.avg:.5f} ' \
                      f'| Tr_Acc ({correct}/{data_size})={accuracy:.2f}% ] '
            logging.info(message)

        self.local_results = {"loss": [], "accuracy": []}
        self.local_results['loss'].append(losses.avg)
        self.local_results['accuracy'].append(accuracy)
