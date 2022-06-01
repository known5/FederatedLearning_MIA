import logging

import torch
import torch.utils
import torch.nn.functional as f
from torch.utils.data import DataLoader

from src.models import AttackModel, get_output_shape_of_last_layer
from .client import Client


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
        self.exploit_gradient = None
        self.exploit_last_layer = None
        self.encoder = None
        self.layers_to_exploit = exploit_layers
        self.gradients_to_exploit = exploit_gradients
        self.exploit_loss = True,
        self.exploit_label = True,
        # self.learning_rate = 0.001
        self.epochs = 1
        self.output_size = int(get_output_shape_of_last_layer(self.model))
        self.one_hot_encoding = create_ohe(self.output_size)

        self.member_attack_train_dataloader = None
        self.non_member_attack_train_dataloader = None
        self.member_attack_test_dataloader = None
        self.non_member_attack_test_dataloader = None

        # Create Attack model based on the target model.
        self.attack_model = AttackModel(target_model=self.model)

    def load_attack_data(self, training_data, split_ratio):
        """ Ja hier moet dus documentatie """
        self.load_data(training_data, split_ratio)

        train_set_size = int(len(training_data) * split_ratio)
        test_set_size = len(training_data) - train_set_size
        train_set, test_set = torch.utils.data.random_split(training_data, [train_set_size, test_set_size])
        member_train_set, non_member_train_set = torch.utils.data.random_split(train_set, [1, 1])
        member_test_set, non_member_test_set = torch.utils.data.random_split(test_set, [1, 1])

        logging.debug(' Loading datasets for attacker and splitting in member and non-member')

        self.member_attack_train_dataloader = DataLoader(member_train_set,
                                                         batch_size=self.batch_size,
                                                         shuffle=True,
                                                         num_workers=2,
                                                         pin_memory=False
                                                         )
        self.non_member_attack_train_dataloader = DataLoader(non_member_train_set,
                                                             batch_size=self.batch_size,
                                                             shuffle=True,
                                                             num_workers=2,
                                                             pin_memory=False
                                                             )

        self.member_attack_test_dataloader = DataLoader(member_test_set,
                                                        batch_size=self.batch_size,
                                                        shuffle=True,
                                                        num_workers=2,
                                                        pin_memory=False
                                                        )
        self.non_member_attack_test_dataloader = DataLoader(non_member_test_set,
                                                            batch_size=self.batch_size,
                                                            shuffle=True,
                                                            num_workers=2,
                                                            pin_memory=False
                                                            )

        logging.debug('loaded attacker data successfully')

    def get_last_layer_outputs(self, features):
        """ Ja hier moet dus documentatie """
        self.model.eval()
        self.model.to(self.device)
        return self.model(features)

    def get_labels(self, labels):
        """
         Returns the labels with one hot encoding.
         """
        return one_hot_encoding(labels, self.one_hot_encoding)

    def get_loss(self, features, labels):
        """
         Computes the loss for given model on given features and labels.
         """
        outputs = self.model(features)
        return self.loss_function(outputs, labels)

    def get_gradients(self, features, labels):
        """ Ja hier moet dus documentatie """
        self.model.train()
        self.model.to(self.device)
        predictions = self.model(features)
        loss = self.loss_function(predictions, labels)
        loss.backward()
        result = None
        for name, param in self.model.named_parameters():
            if self.attack_model.last_layer_name in name:
                result = param.grad
        return result

    def prepare_and_perform_forward_pass(self, features, labels):
        """
            Get all the necessary inputs for a forward pass of the attack model and compute the predictions
        """
        attack_inputs = []
        # Gather and Prepare inputs.
        if self.exploit_last_layer:
            attack_inputs.append(self.get_last_layer_outputs(features))
        if self.exploit_label:
            attack_inputs.append(self.get_labels(labels))
        if self.exploit_loss:
            attack_inputs.append(self.get_loss(features, labels))
        if self.exploit_gradient:
            attack_inputs.append(self.get_gradients(features, labels))
        # Return the predictions of the attack model.
        return self.attack_model(attack_inputs)

    def perform_attack(self):
        """ Ja hier moet dus documentatie """

        for epoch in range(self.attack_epochs):
            self.train_attack()
            if self.eval_attack > 0 and epoch % self.eval_attack == 0:
                self.test_attack()

    def train_attack(self):
        """ Ja hier moet dus documentatie """

        self.attack_model.train()
        self.attack_model.to(self.device)

        for epoch in range(self.number_of_epochs):
            losses = 0

            # for batch_id, (data, labels) in enumerate()

    def test_attack(self):
        """ Ja hier moet dus documentatie """
        pass
