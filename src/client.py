import torch.utils
import torch.optim as optimizers
from torch.utils.data import DataLoader
from src.utils import *


class Client(object):

    def __init__(self, client_id, training_param, device):
        """ Ja hier moet dus documentatie """
        self.loss_function = training_param['loss_function']
        if not hasattr(nn, self.loss_function):
            error_message = f"...Loss Function: \"{self.loss_function}\" is not supported or cannot be found in Torch Optimizers!"
            raise AttributeError(error_message)
        else:
            self.loss_function = nn.__dict__[self.loss_function]()
        self.number_of_epochs = training_param['epochs']

        self.momentum = training_param['momentum']
        self.optimizer = optimizers.__dict__[training_param['optimizer']](
            params=self.model.parameters(),
            lr=training_param['learning_rate'],
            momentum=training_param['momentum'],
            weight_decay=training_param['weight_decay']
        )

        self.batch_size = training_param['batch_size']
        self.client_id = client_id
        self.device = device
        self.data = None
        self.__model = None
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

    def train(self):
        """ Ja hier moet dus documentatie """
        self.model.train()
        self.model.to(self.device)

        for e in range(self.number_of_epochs):
            for data, labels in self.training_dataloader:
                data, labels = data.float().to(self.device), labels.long().to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(data)
                loss = self.loss_function(outputs, labels)

                loss.backward()
                self.optimizer.step()

                if self.device == "cuda": torch.cuda.empty_cache()
        self.model.to("cpu")

    def test(self):
        self.__model.eval()
        self.__model.to(self.device)

        test_loss, correct = 0, 0
        with torch.no_grad():
            for data, labels in self.testing_dataloader:
                data, labels = data.float().to(self.device), labels.long().to(self.device)
                outputs = self.model(data)
                test_loss += self.loss_function(outputs, labels)

                predicted = outputs.argmax(dim=1, keepdim=True)
                correct += predicted.eq(labels.view_as(predicted)).sum().item()
                if self.device == "cuda":
                    torch.cuda.empty_cache()
        self.model.to("cpu")

        self.local_results = {"loss": [], "accuracy": []}
        self.local_results['loss'].append(test_loss / len(self.testing_dataloader))
        self.local_results['accuracy'].append(correct / len(self.testing_dataloader.dataset.indices))
