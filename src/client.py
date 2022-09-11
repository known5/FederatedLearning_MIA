import time
import logging
import torch.utils
import torch.optim as optimizers

from torch.utils.data import DataLoader
from src.utils import AverageMeter, get_torch_loss_function


class Client(object):
    """
     Client Class.

     Used to simulate a client in the Federated Learning network.
     Methods are for loading data, getting and setting the model and
     training and testing on local data with a local model.

     """

    def __init__(self, client_id, training_param, device, model):
        """
        Client constructor method.

        Parameters:
            client_id: Used to identify the client.
            training_param: Training parameters containing setting from the config.yaml file.
            device: Device to be used when doing training and testing (CPU or Cuda GPU).
            model: Copy of the global model used to train on the local data.
        """
        self.client_id = client_id
        self.device = device
        self.__model = model
        # training parameters.
        self.loss_function = get_torch_loss_function(training_param['loss_function'])
        self.batch_size = training_param['train_batch_size']
        # Optimizer parameters.
        self.optimizer_name = training_param['optimizer']
        self.learning_rate = training_param['learning_rate']
        self.momentum = training_param['momentum']
        self.weight_decay = training_param['weight_decay']
        # class shared variables used by methods are initialized here.
        self.training_data = None
        self.test_data = None
        self.training_dataloader = None
        self.testing_dataloader = None
        # scores dictionary.
        self.local_results = {"loss": [], "accuracy": []}

    @property
    def model(self):
        """
         Getter method for client model.
         """
        return self.__model

    @model.setter
    def model(self, model):
        """
         Setter method for client model, also initializes,
         the optimizer with the new model parameters.
         """
        self.optimizer = optimizers.__dict__[self.optimizer_name](
            params=model.parameters(),
            lr=self.learning_rate,
            momentum=self.momentum,
            weight_decay=self.weight_decay
        )
        self.__model = model

    def load_data(self, training_data):
        """
         Method for loading the local data set from the main server.
         Creates a DataLoader object for training and testing.
         """
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
        """
         Training method for the client class.

         Parameters:
             round_number: number of the current epoch to be used for logging.
         """
        # Set model to training mode and transfer to device.
        self.model.train()
        self.model.to(self.device)

        # load once to save time. To be used for calculating the accuracy.
        data_size = len(self.training_dataloader.dataset)

        # Define attributes to keep track of scores during training
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
        """
         Testing method for the client class.

         Parameters:
             round_number: number of the current epoch to be used for logging.
         """
        # Set model to testing mode and transfer to device.
        self.model.eval()
        self.model.to(self.device)

        # load once to save time. To be used for calculating the accuracy.
        data_size = len(self.training_dataloader.dataset)

        # Define attributes to keep track of scores during testing
        batch_time = AverageMeter()
        losses = AverageMeter()
        start_time = time.time()
        correct = 0

        with torch.no_grad():
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
