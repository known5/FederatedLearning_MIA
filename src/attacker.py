import logging
import time

import numpy as np
import torch
import torch.utils
import torch.nn.functional as f
import torch.optim as optimizers
from torch.utils.data import DataLoader

from src.models import AttackModel, get_output_shape_of_last_layer
from .client import Client
from src.utils import AverageMeter, get_torch_loss_function, ConfusionMatrix


def create_ohe(model):
    """ Ja hier moet dus documentatie """
    output_size = get_output_shape_of_last_layer(model)
    return f.one_hot(input=torch.arange(0, output_size), num_classes=output_size)


def one_hot_encoding(labels, encoding, device):
    """ Ja hier moet dus documentatie """
    labels = labels.type(torch.int64).cpu().numpy()
    return torch.stack(list(map(lambda x: encoding[x], labels))).to(device)


class Attacker(Client):

    def __init__(self, client_id, local_data, attack_data, device, target_train_model):
        """
         Attacker class for membership inference attack.
         Based on the KERAS implementation of the paper by Naser et al.

         Paper link: https://arxiv.org/abs/1812.00910

         Implementation link: https://github.com/SPIN-UMass/MembershipWhiteboxAttacks/tree
         /fed4a30107393b42f955ca399c7d0df162e73eb3

         """
        super().__init__(client_id, local_data, device, target_train_model)
        # To exploit layers
        self.exploit_gradient = True
        self.exploit_last_layer = True
        self.exploit_loss = True,
        self.exploit_label = True,

        self.attack_epochs = attack_data['attack_epochs']
        self.eval_attack = attack_data['eval_attack']
        self.attack_batch_size = attack_data['attack_batch_size']
        self.one_hot_encoding = create_ohe(self.model)

        self.attack_data_distribution = attack_data['attack_data_distribution']
        self.attack_data_overlap = attack_data['overlap_with_single_target']
        self.attack_loss_function = get_torch_loss_function(attack_data['attack_loss_function'])
        self.attack_optimizer = attack_data['attack_optimizer']

        # Pre define general variables.
        self.attack_train_member_dataloader = None
        self.attack_train_non_member_dataloader = None
        self.attack_test_member_dataloader = None
        self.attack_test_non_member_dataloader = None

        # Create Attack model based on the target model.
        self.attack_model = AttackModel(target_model=self.model, number_of_classes=200)

    def load_attack_data(self, training_data, test_data):
        """ Ja hier moet dus documentatie """
        logging.debug(' Loading datasets for attacker')
        # Get distributions for attack data from server.
        distribution = self.attack_data_distribution

        # Get the indicis for the right amount of training and test samples.
        training_member_index = torch.randperm(len(training_data))[:distribution[0]]
        training_non_member_index = torch.randperm(len(test_data))[:distribution[1]]
        test_member_index = torch.randperm(len(training_data))[:distribution[2]]
        test_non_member_index = torch.randperm(len(test_data))[:distribution[3]]

        # Create the subsets of the training data necessary.
        train_member_set = torch.utils.data.Subset(training_data, training_member_index)
        train_non_member_set = torch.utils.data.Subset(test_data, training_non_member_index)
        test_member_set = torch.utils.data.Subset(training_data, test_member_index)
        test_non_member_set = torch.utils.data.Subset(test_data, test_non_member_index)

        # Create data loaders
        batch_size = self.attack_batch_size
        self.attack_train_member_dataloader = DataLoader(train_member_set,
                                                         batch_size=batch_size,
                                                         shuffle=True,
                                                         num_workers=2,
                                                         pin_memory=False
                                                         )

        self.attack_train_non_member_dataloader = DataLoader(train_non_member_set,
                                                             batch_size=batch_size,
                                                             shuffle=True,
                                                             num_workers=2,
                                                             pin_memory=False
                                                             )

        self.attack_test_member_dataloader = DataLoader(test_member_set,
                                                        batch_size=batch_size,
                                                        shuffle=True,
                                                        num_workers=2,
                                                        pin_memory=False
                                                        )

        self.attack_test_non_member_dataloader = DataLoader(test_non_member_set,
                                                            batch_size=batch_size,
                                                            shuffle=True,
                                                            num_workers=2,
                                                            pin_memory=False
                                                            )

        logging.debug('loaded attacker data successfully')

    def perform_attack(self, round_number):
        """ Ja hier moet dus documentatie """
        for epoch in range(self.attack_epochs):
            self.train_attack(round_number)
            if self.eval_attack > 0 and epoch % self.eval_attack == 0:
                self.test_attack(round_number)

    def train_attack(self, round_number):
        """ Ja hier moet dus documentatie """
        # Set target model in eval mode and to device
        self.model.eval()

        # Set attack optimizer
        self.attack_optimizer = optimizers.__dict__[self.optimizer_name](
            params=self.model.parameters(),
            lr=self.learning_rate,
            momentum=self.momentum
        )


        data_size = len(self.attack_train_member_dataloader)

        # Perform X number of epochs, each epoch passes the entire dataset once.
        for epoch in range(1, self.attack_epochs + 1):

            confusion_matrix = ConfusionMatrix()
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

                data = torch.cat((member_input, non_member_input)).float().to(self.device)
                labels = torch.cat((member_target, non_member_target)).long().to(self.device)

                self.model.to(self.device)
                predictions = self.model(data)
                if self.exploit_last_layer:
                    attack_inputs.append(predictions)
                if self.exploit_label:
                    one_hot_labels = one_hot_encoding(labels, self.one_hot_encoding, self.device)
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
                acc = np.mean((membership_predictions.data.cpu().numpy() > 0.5) == membership_labels.data.cpu().numpy()) * 100
                confusion_matrix.update(membership_predictions, membership_labels)
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
                    temp = confusion_matrix.get_confusion_matrix()
                    message = f'[ Round: {round_number} ' \
                              f'| Attacker Train ' \
                              f'| Batch: {batch_id * 2}/{data_size} ' \
                              f'| Time: {batch_time.avg:.2f}s ' \
                              f'| Loss: {losses.avg:.5f} ' \
                              f'| Acc: {accuracy.avg:.2f}%' \
                              f'| Conf Matrix: TP:{temp[0]}' \
                              f' FN:{temp[1]}' \
                              f' FP:{temp[2]}' \
                              f' TN:{temp[3]} ]'
                    logging.info(message)
        self.model.to("cpu")

    def test_attack(self, round_number):
        """ Ja hier moet dus documentatie """
        # Set target model in eval mode and to device
        self.model.eval()
        self.model.to(self.device)

        # Perform X number of epochs, each epoch passes the entire dataset once.
        for epoch in range(1 , self.attack_epochs + 1):

            confusion_matrix = ConfusionMatrix()
            batch_time = AverageMeter()
            losses = AverageMeter()
            start_time = time.time()
            accuracy = AverageMeter()

            for batch_id, ((member_input, member_target), (non_member_input, non_member_target)) in enumerate(
                    zip(self.attack_test_member_dataloader
                        , self.attack_test_non_member_dataloader)):
                # Set and updates variables
                attack_inputs = []
                one_hot_labels = None
                gradients = torch.zeros(0)
                # Load data to device

                data = torch.cat((member_input, non_member_input)).float().to(self.device)
                labels = torch.cat((member_target, non_member_target)).long().to(self.device)

                self.model.to(self.device)
                predictions = self.model(data)
                if self.exploit_last_layer:
                    attack_inputs.append(predictions)
                if self.exploit_label:
                    one_hot_labels = one_hot_encoding(labels, self.one_hot_encoding, self.device)
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
                self.attack_model.eval()
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
                acc = np.mean((membership_predictions.data.cpu().numpy() > 0.5) == membership_labels.data.cpu().numpy()) * 100
                confusion_matrix.update(membership_predictions, membership_labels)
                accuracy.update(acc, self.attack_batch_size)
                losses.update(attack_loss.item(), self.attack_batch_size)
                batch_time.update(time.time() - start_time)

                self.attack_model.to('cpu')

                if batch_id % 10 == 0:
                    temp = confusion_matrix.get_confusion_matrix()
                    message = f'[ Round: {round_number} ' \
                              f'| Attacker Test ' \
                              f'| Batch: {batch_id}/{len(self.attack_test_member_dataloader)} ' \
                              f'| Time: {batch_time.avg:.2f}s ' \
                              f'| Loss: {losses.avg:.5f} ' \
                              f'| Acc: {accuracy.avg:.2f}% ]' \
                              f'| Conf Matrix: TP:{temp[0]}' \
                              f' FN:{temp[1]}' \
                              f' FP:{temp[2]}' \
                              f' TN:{temp[3]} ]'
                    logging.info(message)
        self.model.to("cpu")
