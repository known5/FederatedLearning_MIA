import logging
import time

import numpy as np
import torch
import torch.utils
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optimizers
from torch.utils.data import DataLoader

from src.models import AttackModel, get_output_shape_of_last_layer
from .client import Client
from src.utils import AverageMeter


def create_ohe(output_size):
    """ Ja hier moet dus documentatie """
    return f.one_hot(input=torch.arange(0, output_size), num_classes=output_size)


def one_hot_encoding(labels, encoding):
    """ Ja hier moet dus documentatie """
    labels = labels.type(torch.int64).numpy()
    return torch.stack(list(map(lambda x: encoding[x], labels)))


def split(x):
    """ Ja hier moet dus documentatie """
    return torch.split(x, len(x.numpy()))


class Attacker(Client):

    def __init__(self, client_id, local_data, device, target_train_model, exploit_layers, exploit_gradients):
        """
         Attacker class for membership inference attack.
         Based on the KERAS implementation of the paper by Naser et al.

         Paper link: https://arxiv.org/abs/1812.00910

         Implementation link: https://github.com/privacytrustlab/ml_privacy_meter/tree/master/archive

         """
        super().__init__(client_id, local_data, device, target_train_model)
        self.attack_epochs = 1
        self.eval_attack = 0
        self.exploit_gradient = True
        self.exploit_last_layer = True
        self.encoder = None
        self.exploit_loss = True,
        self.exploit_label = True,
        # self.learning_rate = 0.001
        self.epochs = 1
        self.attack_batch_size = 64
        self.output_size = int(get_output_shape_of_last_layer(self.model))
        self.one_hot_encoding = create_ohe(self.output_size)

        self.attack_train_member_dataloader = None
        self.attack_train_non_member_dataloader = None
        self.attack_test_member_dataloader = None
        self.attack_test_non_member_dataloader = None
        self.attack_test_dataloader = None
        self.attack_loss_function = local_data['attack_loss_function']
        if not hasattr(nn, self.attack_loss_function):
            error_message = f"...Loss Function: \"{self.attack_loss_function}\" is not supported or cannot be found in " \
                            f"Torch Optimizers! "
            logging.error(error_message)
            raise AttributeError(error_message)
        else:
            self.attack_loss_function = nn.__dict__[self.attack_loss_function]()
        self.attack_optimizer = local_data['attack_optimizer']

        # Create Attack model based on the target model.
        self.attack_model = AttackModel(target_model=self.model, number_of_classes=200)

    def load_attack_data(self, training_data, test_data, attack_data_distribution):
        """ Ja hier moet dus documentatie """
        logging.debug(' Loading datasets for attacker')

        # Get the indicis for the right amount of training and test samples.
        training_member_index = torch.randperm(len(training_data))[:attack_data_distribution[0]]
        training_non_member_index = torch.randperm(len(test_data))[:attack_data_distribution[1]]
        test_member_index = torch.randperm(len(training_data))[:attack_data_distribution[2]]
        test_non_member_index = torch.randperm(len(test_data))[:attack_data_distribution[3]]

        # Create the subsets of the training data necessary.
        train_member_set = torch.utils.data.Subset(training_data, training_member_index)
        train_non_member_set = torch.utils.data.Subset(test_data, training_non_member_index)
        test_member_set = torch.utils.data.Subset(training_data, test_member_index)
        test_non_member_set = torch.utils.data.Subset(test_data, test_non_member_index)

        # Create data loaders
        self.attack_train_member_dataloader = DataLoader(train_member_set,
                                                         batch_size=self.batch_size,
                                                         shuffle=True,
                                                         num_workers=2,
                                                         pin_memory=False
                                                         )

        self.attack_train_non_member_dataloader = DataLoader(train_non_member_set,
                                                             batch_size=self.batch_size,
                                                             shuffle=True,
                                                             num_workers=2,
                                                             pin_memory=False
                                                             )

        self.attack_test_member_dataloader = DataLoader(test_member_set,
                                                        batch_size=self.batch_size,
                                                        shuffle=True,
                                                        num_workers=2,
                                                        pin_memory=False
                                                        )

        self.attack_test_non_member_dataloader = DataLoader(test_non_member_set,
                                                            batch_size=self.batch_size,
                                                            shuffle=True,
                                                            num_workers=2,
                                                            pin_memory=False
                                                            )

        logging.debug('loaded attacker data successfully')

    def perform_attack(self):
        """ Ja hier moet dus documentatie """
        for epoch in range(self.attack_epochs):
            self.train_attack()
            if self.eval_attack > 0 and epoch % self.eval_attack == 0:
                self.test_attack()

    def train_attack(self, round_number):
        """ Ja hier moet dus documentatie """
        # Set target model in eval mode and to device
        self.model.eval()
        self.model.to(self.device)

        # Set attack optimizer
        self.attack_optimizer = optimizers.__dict__[self.optimizer_name](
            params=self.model.parameters(),
            lr=self.learning_rate,
            momentum=self.momentum
        )

        # Perform X number of epochs, each epoch passes the entire dataset once.
        for epoch in range(self.number_of_epochs):
            epoch += 1

            batch_time = AverageMeter()
            losses = AverageMeter()
            start_time = time.time()
            accuracy = AverageMeter()

            for batch_id, ((member_input, member_target), (non_member_input, non_member_target)) in enumerate(
                    zip(self.attack_train_member_dataloader
                        , self.attack_train_non_member_dataloader)):
                # Set and updates variables
                attack_inputs = []
                one_hot_labels = None
                gradients = torch.zeros(0)
                # Load data to device
                member_input, member_target = member_input.float().to(self.device) \
                    , member_target.long().to(self.device)
                non_member_input, non_member_target = non_member_input.float().to(self.device) \
                    , non_member_target.long().to(self.device)

                data, labels = torch.cat((member_input, non_member_input)), torch.cat(
                    (member_target, non_member_target))

                predictions = self.model(data)
                if self.exploit_last_layer:
                    attack_inputs.append(predictions)
                if self.exploit_label:
                    one_hot_labels = one_hot_encoding(labels, self.one_hot_encoding)
                    attack_inputs.append(one_hot_labels.float())
                if self.exploit_loss:
                    temp = torch.sum(predictions * one_hot_labels, dim=1).view([-1, 1])
                    attack_inputs.append(temp)
                if self.exploit_gradient:
                    for index in range(predictions.size(0)):
                        loss = self.loss_function(predictions[index].view([1, -1]), labels[index].view([-1]))
                        self.optimizer.zero_grad()
                        if index == (predictions.size(0)) - 1:
                            loss.backward(retain_graph=False)
                        else:
                            loss.backward(retain_graph=True)
                        current_grads = self.model.classifier.weight.grad.view([1, 1, 256, 200])

                        if gradients.size()[0] == 0:
                            gradients = current_grads
                        else:
                            gradients = torch.cat((gradients, current_grads))
                    attack_inputs.append(gradients)

                # Get the predictions for the membership classification
                self.model.to("cpu")
                self.attack_model.train()
                self.attack_model.to(self.device)

                # Set all data components to device
                for component in range(len(attack_inputs)):
                    attack_inputs[component] = attack_inputs[component].data.float().to(self.device)
                membership_predictions = self.attack_model(attack_inputs)

                # Change labels of the data for binary attack classification
                member_target = torch.Tensor([1 for _ in member_target])
                non_member_target = torch.Tensor([0 for _ in non_member_target])
                membership_labels = torch.cat((member_target, non_member_target))
                membership_labels = torch.unsqueeze(membership_labels, -1).data.float().to(self.device)

                # Calculate the loss of attack model
                attack_loss = self.attack_loss_function(membership_predictions, membership_labels)

                # Measure training accuracy and report metrics
                acc = np.mean((membership_predictions.data.numpy() > 0.5) == membership_labels.data.numpy())
                accuracy.update(acc, self.attack_batch_size)
                losses.update(attack_loss.item(), self.attack_batch_size)
                batch_time.update(time.time() - start_time)

                # Perform optimizer steps
                self.attack_optimizer.zero_grad()
                attack_loss.backward()
                self.attack_optimizer.step()
                # Place model back to CPU
                self.attack_model.to('cpu')

                if batch_id % 10 == 0:
                    message = f'[ Round: {round_number} ' \
                              f'| Attacker Train ' \
                              f'| Batch: {batch_id}/{len(self.attack_train_member_dataloader)} ' \
                              f'| Time: {batch_time.avg:.2f}s ' \
                              f'| Loss: {losses.avg:.5f} ' \
                              f'| TR_ACC: {accuracy.avg:.2f}% ]'
                    logging.info(message)
        self.model.to("cpu")

    def test_attack(self):
        """ Ja hier moet dus documentatie """
        pass
