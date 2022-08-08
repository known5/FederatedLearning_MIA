import copy
import logging
import random
import time

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
        self.attack_data_loader = None
        self.target_model = None
        self.active_attack_optimizer = None
        self.active_learning_rate = 0.0001

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

        # check is data sizes are correct:
        for length in distribution:
            assert length // 100 != 0 and length % 100 == 0

        train_member_length = distribution[0] // 100
        train_non_member_length = distribution[1] // 100
        test_member_length = distribution[2] // 100
        test_non_member_length = distribution[3] // 100

        for target in range(self.number_of_classes):
            # Set index lists to empty list.
            training_member_index, training_non_member_index, test_member_index, test_non_member_index = [], [], [], []
            # Set the boundaries for picking samples from the right classes
            low_member, high_member = target * 1000, (target + 1) * 1000
            low_non_member, high_non_member = target * 100, (target + 1) * 100

            random_subset = list(range(low_member, high_member))
            random.shuffle(random_subset)

            for _ in range(train_member_length):
                training_member_index.append(random_subset.pop())

            for _ in range(test_member_length):
                test_member_index.append(random_subset.pop())

            random_subset = list(range(low_non_member, high_non_member))
            random.shuffle(random_subset)

            for _ in range(train_non_member_length):
                training_non_member_index.append(random_subset.pop())

            for _ in range(test_non_member_length):
                test_non_member_index.append(random_subset.pop())

            # Create the subsets for this class
            train_member_set = torch.utils.data.Subset(training_data, training_member_index)
            train_non_member_set = torch.utils.data.Subset(non_member_data, training_non_member_index)
            test_member_set = torch.utils.data.Subset(training_data, test_member_index)
            test_non_member_set = torch.utils.data.Subset(non_member_data, test_non_member_index)

            self.class_attack_data_subsets.append((train_member_set, train_non_member_set))
            self.class_test_data_subsets.append((test_member_set, test_non_member_set))

        logging.debug('loaded attacker data successfully')

    def gradient_ascent_attack(self, round_number):
        """ Ja hier moet dus documentatie """
        # Put model in training mode and load model onto device

        self.target_model = copy.deepcopy(self.model)
        self.active_attack_optimizer = optimizers.__dict__['SGD'](
            params=self.target_model.parameters(),
            maximize=True,
            lr=self.active_learning_rate
        )
        self.target_model.train()
        self.target_model.to(self.device)

        for data_class in range(self.number_of_classes):
            # Setup variables to log statistics.
            losses = 0
            batch_time = AverageMeter()
            loss_meter = AverageMeter()
            start_time = time.time()

            member, non_member = self.class_attack_data_subsets[data_class]
            active_set = torch.utils.data.ConcatDataset([member, non_member])
            attack_data_loader = DataLoader(active_set,
                                            batch_size=self.attack_batch_size,
                                            shuffle=True,
                                            num_workers=2,
                                            pin_memory=True
                                            )
            for data, labels in attack_data_loader:
                # Transfer data to CPU or GPU and set gradients to zero for performance.
                data, labels = data.float().to(self.device), labels.long().to(self.device)
                self.active_attack_optimizer.zero_grad()

                # Do a forward pass through the network to get prediction values and update loss metric.
                outputs = self.target_model(data)
                loss = self.loss_function(outputs, labels)

                # Do a backward pass through the network to get the gradients
                # and then use the optimizer to update the weights.
                loss.backward()
                self.active_attack_optimizer.step()
                losses += loss.item()

            # Update loss and run_time metrics
            loss_meter.update(losses)
            batch_time.update(time.time() - start_time)

            message = f'[ Round: {round_number} ' \
                      f'| Active Attack ' \
                      f'| Class: {data_class}' \
                      f'| Client: {self.client_id} ' \
                      f'| Time: {batch_time.avg:.2f}s ' \
                      f'| Loss: {loss_meter.avg:.5f} '
            logging.info(message)
        self.target_model.to("cpu")
        self.model = copy.deepcopy(self.target_model)

    def train_attack(self, round_number, target_models):
        """ Ja hier moet dus documentatie """
        # confusion matrix and accuracy metric initialized
        start_time = time.time()
        final_confusion_matrix = ConfusionMatrix()
        class_confusion_matrix = ConfusionMatrix()
        losses = AverageMeter()
        accuracy = AverageMeter()
        class_time = AverageMeter()

        self.attack_model.train()

        # perform inference per class for confusion matrix to keep accurate track of scores.
        for data_class in range(self.number_of_classes):
            class_start_time = time.time()
            class_time.reset()
            class_confusion_matrix.reset()
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

                # Get the predictions for the membership classification
                self.attack_model.to(self.device)

                # Change labels of the data for binary attack classification
                member_target = torch.Tensor([1 for _ in member_input])
                non_member_target = torch.Tensor([0 for _ in non_member_input])
                membership_labels = torch.cat((member_target, non_member_target))
                membership_labels = torch.unsqueeze(membership_labels, -1).data.float().to(self.device)

                # Get membership predictions
                membership_predictions = self.attack_model(model_outputs, one_hot_labels, loss_values, model_gradients)
                # Calculate the loss of attack model
                attack_loss = self.attack_loss_function(membership_predictions, membership_labels)

                # Perform optimizer steps
                self.attack_optimizer.zero_grad()
                attack_loss.backward()
                self.attack_optimizer.step()
                # Place model back to CPU
                self.attack_model.to('cpu')

                # Measure training accuracy and report metric
                class_confusion_matrix.update(membership_predictions, membership_labels)
                accuracy.update(class_confusion_matrix.get_accuracy(), self.attack_batch_size)
                losses.update(attack_loss.item(), self.attack_batch_size)
                class_time.update(time.time() - class_start_time)

            temp = class_confusion_matrix.get_confusion_matrix()
            final_confusion_matrix.update_from_matrix(temp)
            message = f'[ Round: {round_number} ' \
                      f'| Attacker Train ' \
                      f'| Class: {data_class}' \
                      f'| Time: {class_time.avg:.2f}s ' \
                      f'| Loss: {losses.avg:.5f} ' \
                      f'| Avg Acc: {accuracy.avg:.2f}% ]' \
                      f'| Class Conf Matrix: TP:{temp[0]}' \
                      f' FP:{temp[1]}' \
                      f' TN:{temp[2]}' \
                      f' FN:{temp[3]} ]'
            logging.info(message)

        temp = final_confusion_matrix.get_confusion_matrix()
        message = f'[ Round: {round_number} Totals ' \
                  f'| Attacker Train ' \
                  f'| Total Time: {time.time() - start_time:.2f}s ' \
                  f'| Loss: {losses.avg:.5f} ' \
                  f'| Acc: {final_confusion_matrix.get_accuracy():.2f}% ]' \
                  f'| Final Conf Matrix: TP:{temp[0]}' \
                  f' FP:{temp[1]}' \
                  f' TN:{temp[2]}' \
                  f' FN:{temp[3]} ]'
        logging.info(message)

    def test_attack(self, round_number, target_models):
        """ Ja hier moet dus documentatie """
        # Metrics
        start_time = time.time()
        final_confusion_matrix = ConfusionMatrix()
        class_confusion_matrix = ConfusionMatrix()
        losses = AverageMeter()
        accuracy = AverageMeter()
        class_time = AverageMeter()

        self.attack_model.eval()

        for data_class in range(self.number_of_classes):
            class_start_time = time.time()
            class_time.reset()
            class_confusion_matrix.reset()
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
                # Load data to device
                member_input, member_target = member_input.float().to(self.device) \
                    , member_target.long().to(self.device)
                non_member_input, non_member_target = non_member_input.float().to(self.device) \
                    , non_member_target.long().to(self.device)

                data, labels = torch.cat((member_input, non_member_input)), torch.cat(
                    (member_target, non_member_target))

                # Create one-hot encoding of labels, as this has to be done only once.
                one_hot_labels = one_hot_encoding(labels, self.one_hot_encoding, self.device).float()

                model_outputs = []
                loss_values = []
                model_gradients = []

                #
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

                # Get the predictions for the membership classification
                self.attack_model.to(self.device)
                with torch.no_grad():
                    # Get membership predictions
                    membership_predictions = self.attack_model(model_outputs, one_hot_labels, loss_values,
                                                               model_gradients)

                    # Change labels of the data for binary attack classification
                    member_target = torch.Tensor([1 for _ in member_target])
                    non_member_target = torch.Tensor([0 for _ in non_member_target])
                    membership_labels = torch.cat((member_target, non_member_target))
                    membership_labels = torch.unsqueeze(membership_labels, -1).data.float().to(self.device)

                    # Calculate the loss of attack model
                    attack_loss = self.attack_loss_function(membership_predictions, membership_labels)

                    class_confusion_matrix.update(membership_predictions, membership_labels)
                    accuracy.update(class_confusion_matrix.get_accuracy(), self.attack_batch_size)
                    losses.update(attack_loss.item(), self.attack_batch_size)
                    class_time.update(time.time() - class_start_time)

                self.attack_model.to('cpu')

            temp = class_confusion_matrix.get_confusion_matrix()
            final_confusion_matrix.update_from_matrix(temp)
            message = f'[ Round: {round_number} ' \
                      f'| Attacker Test ' \
                      f'| Class: {data_class}' \
                      f'| Time: {class_time.avg:.2f}s ' \
                      f'| Loss: {losses.avg:.5f} ' \
                      f'| Avg Acc: {accuracy.avg:.2f}% ]' \
                      f'| Class Conf Matrix: TP:{temp[0]}' \
                      f' FP:{temp[1]}' \
                      f' TN:{temp[2]}' \
                      f' FN:{temp[3]} ]'
            logging.info(message)

        temp = final_confusion_matrix.get_confusion_matrix()
        message = f'[ Round: {round_number} Totals ' \
                  f'| Attacker Test ' \
                  f'| Total Time: {time.time() - start_time:.2f}s ' \
                  f'| Loss: {losses.avg:.5f} ' \
                  f'| Acc: {final_confusion_matrix.get_accuracy():.2f}% ]' \
                  f'| Final Conf Matrix: TP:{temp[0]}' \
                  f' FP:{temp[1]}' \
                  f' TN:{temp[2]}' \
                  f' FN:{temp[3]} ]'
        logging.info(message)
