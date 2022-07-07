import logging
import time

import numpy as np
import torch
import torch.nn.functional as f
import torch.optim as optimizers
import torch.utils
from torch.utils.data import DataLoader

from src.models import AttackModel, get_output_shape_of_last_layer
from src.utils import AverageMeter, get_torch_loss_function, ConfusionMatrix
from .client import Client


def create_ohe(model):
    """ Ja hier moet dus documentatie """
    output_size = get_output_shape_of_last_layer(model)
    return f.one_hot(input=torch.arange(0, output_size), num_classes=output_size)


def one_hot_encoding(labels, encoding, device):
    """ Ja hier moet dus documentatie """
    labels = labels.type(torch.int64).cpu().numpy()
    return torch.stack(list(map(lambda x: encoding[x], labels))).to(device)


class Attacker(Client):

    def __init__(self, client_id, local_data, attack_data, device, target_train_model, number_of_observed_models):
        """
         Attacker class for membership inference attack.
         Based on the KERAS implementation of the paper by Naser et al.

         Paper link: https://arxiv.org/abs/1812.00910

         Implementation link: https://github.com/SPIN-UMass/MembershipWhiteboxAttacks/tree
         /fed4a30107393b42f955ca399c7d0df162e73eb3

         """
        super().__init__(client_id, local_data, device, target_train_model)
        self.attack_optimizer = None
        self.eval_attack = attack_data['eval_attack']
        self.attack_batch_size = attack_data['attack_batch_size']
        self.one_hot_encoding = create_ohe(self.model)

        self.attack_data_distribution = attack_data['attack_data_distribution']
        self.attack_loss_function = get_torch_loss_function(attack_data['attack_loss_function'])
        self.attack_optimizer_name = attack_data['attack_optimizer']

        # Pre define general variables.
        self.class_attack_data_subsets = []
        self.class_test_data_subsets = []
        self.number_of_classes = 100

        # Create Attack model based on the target model.
        self.attack_model = AttackModel(target_model=self.model,
                                        number_of_classes=self.number_of_classes,
                                        number_of_observed_models=number_of_observed_models)

        # Set attack optimizer after sending model to device
        self.attack_optimizer = optimizers.__dict__[self.attack_optimizer_name](
            params=self.attack_model.parameters(),
            lr=0.0001
        )

    def load_attack_data(self, training_data, non_member_data):
        """ Ja hier moet dus documentatie """
        logging.debug(' Loading datasets for attacker')
        # Get distributions for attack data from server.
        distribution = self.attack_data_distribution
        # Get the indicis for the right amount of training and test samples.
        self.class_attack_data_subsets = []
        self.class_test_data_subsets = []

        message = f'[ Attack data distribution: {distribution} ]'
        logging.info(message)

        training_data_remainder = len(training_data) - (distribution[0] + distribution[2])
        length_split = [distribution[0], distribution[2], training_data_remainder]
        member_data = torch.utils.data.random_split(training_data, length_split)

        for target in range(self.number_of_classes):
            # Set the boundaries for picking samples from the right classes
            low, high = target * 100, (target + 1) * 100

            #
            training_member_index = [ind for ind, (x, y) in enumerate(member_data[0]) if y == target]
            training_non_member_index = torch.randint(low=low,
                                                      high=high,
                                                      size=(1, len(training_member_index)))[0].tolist()

            #
            test_member_index = [ind for ind, (x, y) in enumerate(member_data[1]) if y == target]
            test_non_member_index = torch.randint(low=low,
                                                  high=high,
                                                  size=(1, len(test_member_index)))[0].tolist()

            # Create the subsets for this class
            train_member_set = torch.utils.data.Subset(member_data[0], training_member_index)
            train_non_member_set = torch.utils.data.Subset(non_member_data, training_non_member_index)
            test_member_set = torch.utils.data.Subset(member_data[1], test_member_index)
            test_non_member_set = torch.utils.data.Subset(non_member_data, test_non_member_index)

            self.class_attack_data_subsets.append((train_member_set, train_non_member_set))
            self.class_test_data_subsets.append((test_member_set, test_non_member_set))

        logging.debug('loaded attacker data successfully')

    def train_attack(self, round_number, target_models):
        """ Ja hier moet dus documentatie """
        # confusion matrix and accuracy metric initialized
        confusion_matrix = ConfusionMatrix()
        accuracy = AverageMeter()
        batch_time = AverageMeter()
        losses = AverageMeter()
        # perform inference per class for confusion matrix to keep accurate track of scores.
        for data_class in range(self.number_of_classes):
            start_time = time.time()
            confusion_matrix.reset()
            member, non_member = self.class_attack_data_subsets[data_class]

            temp_data_loader = DataLoader(member,
                                          batch_size=self.attack_batch_size,
                                          shuffle=True,
                                          num_workers=2,
                                          pin_memory=True
                                          )

            temp_data_loader_2 = DataLoader(non_member,
                                            batch_size=self.attack_batch_size,
                                            shuffle=True,
                                            num_workers=2,
                                            pin_memory=True
                                            )

            data_loader = zip(temp_data_loader, temp_data_loader_2)
            for (member_input, member_target), (non_member_input, non_member_target) in data_loader:
                # Pre-define variables to use.
                model_outputs = []
                loss_values = []
                model_gradients = []

                # Load data to device
                member_input, member_target = member_input.float().to(self.device) \
                    , member_target.long().to(self.device)
                non_member_input, non_member_target = non_member_input.float().to(self.device) \
                    , non_member_target.long().to(self.device)

                data, labels = torch.cat((member_input, non_member_input)), torch.cat(
                    (member_target, non_member_target))

                # Create one-hot encoding of labels, as this has to be done only once.
                one_hot_labels = one_hot_encoding(labels, self.one_hot_encoding, self.device).float()

                # For each observed model collect input data for the attack model.
                for model in target_models:
                    #
                    model.eval()
                    model.to(self.device)
                    optimizer = optimizers.__dict__[self.optimizer_name](
                        params=model.parameters(),
                        lr=self.learning_rate,
                        momentum=self.momentum,
                        weight_decay=self.weight_decay
                    )

                    # Get predictions
                    predictions = model(data)
                    temp_pred = torch.from_numpy(predictions.data.cpu().numpy()).to(self.device)
                    model_outputs.append(temp_pred.requires_grad_(True))
                    # calculate loss values
                    loss_value = torch.sum(predictions * one_hot_labels, dim=1).view([-1, 1])
                    temp_loss = torch.from_numpy(loss_value.data.cpu().numpy()).to(self.device)
                    loss_values.append(temp_loss.requires_grad_(True))

                    gradients = torch.zeros(0)
                    for index in range(predictions.size(0)):
                        loss = self.loss_function(predictions[index].view([1, -1]), labels[index].view([-1]))
                        optimizer.zero_grad()
                        if index == (predictions.size(0)) - 1:
                            loss.backward(retain_graph=False)
                        else:
                            loss.backward(retain_graph=True)
                        current_grads = model.classifier.weight.grad.view([1, 1, 256, self.number_of_classes])

                        if gradients.size()[0] == 0:
                            gradients = current_grads
                        else:
                            gradients = torch.cat((gradients, current_grads))

                    temp_gradients = torch.from_numpy(gradients.data.cpu().numpy()).to(self.device)
                    model_gradients.append(temp_gradients.requires_grad_(True))
                    model.to('cpu')
                # remove data from GPU to free up memory
                data.to('cpu')

                # Get the predictions for the membership classification
                self.attack_model.train()
                self.attack_model.to(self.device)

                # Change labels of the data for binary attack classification
                member_target = torch.Tensor([1 for _ in member_input])
                non_member_target = torch.Tensor([0 for _ in non_member_input])
                membership_labels = torch.cat((member_target, non_member_target))
                membership_labels = torch.unsqueeze(membership_labels, -1).data.float().to(self.device)

                # Get membership predictions
                membership_predictions = self.attack_model(model_outputs, one_hot_labels, loss_values, model_gradients)
                print(membership_predictions)
                # Calculate the loss of attack model
                attack_loss = self.attack_loss_function(membership_predictions, membership_labels)

                # Perform optimizer steps
                self.attack_optimizer.zero_grad()
                attack_loss.backward()
                self.attack_optimizer.step()
                # Place model back to CPU
                self.attack_model.to('cpu')

                # Measure training accuracy and report metric
                acc = np.mean(
                    (membership_predictions.data.cpu().numpy() > 0.5) == membership_labels.data.cpu().numpy()) * 100
                confusion_matrix.update(membership_predictions, membership_labels)
                accuracy.update(acc, self.attack_batch_size)
                losses.update(attack_loss.item(), self.attack_batch_size)
                batch_time.update(time.time() - start_time)

            temp = confusion_matrix.get_confusion_matrix()
            message = f'[ Round: {round_number} ' \
                      f'| Attacker Train ' \
                      f'| Class: {data_class}' \
                      f'| Time: {batch_time.avg:.2f}s ' \
                      f'| Loss: {losses.avg:.5f} ' \
                      f'| Acc: {accuracy.avg:.2f}% ]' \
                      f'| Conf Matrix: TP:{temp[0]}' \
                      f' FP:{temp[1]}' \
                      f' TN:{temp[2]}' \
                      f' FN:{temp[3]} ]'
            logging.info(message)
        self.model.to("cpu")

    def test_attack(self, round_number, target_models):
        """ Ja hier moet dus documentatie """
        # Set target model in eval mode and to device
        self.model.eval()
        self.model.to(self.device)

        # Metrics
        confusion_matrix = ConfusionMatrix()
        batch_time = AverageMeter()
        losses = AverageMeter()
        accuracy = AverageMeter()
        for data_class in range(self.number_of_classes):
            start_time = time.time()
            confusion_matrix.reset()
            member, non_member = self.class_test_data_subsets[data_class]

            temp_data_loader = DataLoader(member,
                                          batch_size=self.attack_batch_size,
                                          shuffle=True,
                                          num_workers=2,
                                          pin_memory=False
                                          )

            temp_data_loader_2 = DataLoader(non_member,
                                            batch_size=self.attack_batch_size,
                                            shuffle=True,
                                            num_workers=2,
                                            pin_memory=False
                                            )

            data_loader = zip(temp_data_loader, temp_data_loader_2)
            for (member_input, member_target), (non_member_input, non_member_target) in data_loader:
                gradients = torch.zeros(0)
                # Load data to device
                member_input, member_target = member_input.float().to(self.device) \
                    , member_target.long().to(self.device)
                non_member_input, non_member_target = non_member_input.float().to(self.device) \
                    , non_member_target.long().to(self.device)

                data, labels = torch.cat((member_input, non_member_input)), torch.cat(
                    (member_target, non_member_target))

                # Create one-hot encoding of labels, as this has to be done only once.
                one_hot_labels = one_hot_encoding(labels, self.one_hot_encoding, self.device)

                model_outputs = []
                loss_values = []
                model_gradients = []

                #
                for model in target_models:
                    #
                    model.eval()
                    model.to(self.device)
                    self.optimizer = optimizers.__dict__[self.optimizer_name](
                        params=self.model.parameters(),
                        lr=self.learning_rate,
                        momentum=self.momentum,
                        weight_decay=self.weight_decay
                    )

                    # Get predictions
                    predictions = model(data)
                    model_outputs.append(predictions)
                    # calculate loss values
                    loss_value = torch.sum(predictions * one_hot_labels, dim=1).view([-1, 1])
                    loss_values.append(loss_value)

                    #
                    gradients = torch.zeros(0)
                    for index in range(predictions.size(0)):
                        loss = self.loss_function(predictions[index].view([1, -1]), labels[index].view([-1]))
                        self.optimizer.zero_grad()
                        if index == (predictions.size(0)) - 1:
                            loss.backward(retain_graph=False)
                        else:
                            loss.backward(retain_graph=True)
                        current_grads = model.classifier.weight.grad.view([1, 1, 256, self.number_of_classes])

                        if gradients.size()[0] == 0:
                            gradients = current_grads
                        else:
                            gradients = torch.cat((gradients, current_grads))
                    model_gradients.append(gradients)
                    model.to('cpu')

                # remove data from GPU to free up memory
                data.to('cpu')

                # Get the predictions for the membership classification
                self.attack_model.train()
                self.attack_model.to(self.device)

                # Set attack optimizer after sending model to device
                self.attack_optimizer = optimizers.__dict__[self.optimizer_name](
                    params=self.model.parameters(),
                    lr=self.learning_rate,
                    momentum=self.momentum
                )

                self.attack_optimizer.zero_grad()

                # Set all data components to device
                for component in range(len(target_models)):
                    model_outputs[component] = model_outputs[component].float().to(self.device)
                    loss_values[component] = loss_values[component].float().to(self.device)
                    model_gradients[component] = model_gradients[component].float().to(self.device)
                one_hot_labels = one_hot_labels.float().to(self.device)

                # Get membership predictions
                membership_predictions = self.attack_model(model_outputs, one_hot_labels, loss_values, model_gradients)

                # Change labels of the data for binary attack classification
                member_target = torch.Tensor([1 for _ in member_target])
                non_member_target = torch.Tensor([0 for _ in non_member_target])
                membership_labels = torch.cat((member_target, non_member_target))
                membership_labels = torch.unsqueeze(membership_labels, -1).data.float().to(self.device)

                # Calculate the loss of attack model
                attack_loss = self.attack_loss_function(membership_predictions, membership_labels)

                # Measure training accuracy and report metrics
                acc = np.mean(
                    (membership_predictions.data.cpu().numpy() > 0.5) == membership_labels.data.cpu().numpy()) * 100
                confusion_matrix.update(membership_predictions, membership_labels)
                accuracy.update(acc, self.attack_batch_size)
                losses.update(attack_loss.item(), self.attack_batch_size)
                batch_time.update(time.time() - start_time)

                self.attack_model.to('cpu')

            temp = confusion_matrix.get_confusion_matrix()
            message = f'[ Round: {round_number} ' \
                      f'| Attacker Test ' \
                      f'| Class: {data_class}' \
                      f'| Time: {batch_time.avg:.2f}s ' \
                      f'| Loss: {losses.avg:.5f} ' \
                      f'| Acc: {accuracy.avg:.2f}% ]' \
                      f'| Conf Matrix: TP:{temp[0]}' \
                      f' FP:{temp[1]}' \
                      f' TN:{temp[2]}' \
                      f' FN:{temp[3]} ]'
            logging.info(message)
        self.model.to("cpu")
