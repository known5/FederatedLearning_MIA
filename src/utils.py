import os
import time
import shutil
import logging

import torch
import torch.nn as nn
import torch.nn.functional as f
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from src.models import AlexNet
from src.models import get_output_shape_of_last_layer


########################################################################################################################
# Helper methods
########################################################################################################################

def get_torch_loss_function(name_of_loss_function):
    """
     Loads the required loss function object from the torch.nn library.
     If the library does not have the required loss function it throws an AttributeError.

     Parameters:
         name_of_loss_function: name of the required loss function.
     """
    if not hasattr(nn, name_of_loss_function):
        error_message = f"...Loss Function: \"{name_of_loss_function}\" is not supported or cannot be found in " \
                        f"Torch Optimizers! "
        logging.error(error_message)
        raise AttributeError(error_message)
    else:
        return nn.__dict__[name_of_loss_function]()


def create_one_hot_encoding(model):
    """
        Creates a one-hot encoding object based on the given model.

        Parameters:
            model: model to base the one hot encoding on.
     """
    output_size = get_output_shape_of_last_layer(model)
    return f.one_hot(input=torch.arange(0, output_size), num_classes=output_size)


def encode_labels(labels, encoding, device):
    """
     Encodes the given labels according to the given one-hot-encoding.

     Parameters:
         labels: Labels to be encoded.
         encoding: one-hot encoding used for transforming the labels.
         device: Device to simulate the models on.
     """
    labels = labels.type(torch.int64).cpu().numpy()
    return torch.stack(list(map(lambda x: encoding[x], labels))).to(device)


def get_duration(start_time):
    """
     Transforms a time interval format from seconds into hours, minutes and seconds.

     Parameters:
         start_time: time stamp used to define the time interval.
     """
    hours, rem = divmod(time.time() - start_time, 3600)
    minutes, seconds = divmod(rem, 60)
    return "{:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds)


def save_checkpoint(state, is_best, checkpoint='checkpoint', filename='checkpoint.pth.tar'):
    """
    TODO

     """
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth.tar'))


def save_checkpoint_adversary(state, is_best, checkpoint='checkpoint', filename='checkpoint.pth.tar'):
    """ Ja hier moet dus documentatie """
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_adversary_best.pth.tar'))


def load_target_model(path_to_model_checkpoint, filename):
    """
     Loads the target model from a previous checkpoint to resume training or
     to be used to generate input for the attack model.

     Note this has hardcoded in to load the AlexNet model from src.models.

     Parameters:
         path_to_model_checkpoint: path to the location of the model checkpoint.
         filename: name of the model checkpoint file.
     """
    model = AlexNet()
    filepath = os.path.join(path_to_model_checkpoint, filename)
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['state_dict'])
    return model


def load_dataset(path_for_data_set, data_set_name):
    """
     Main method for loading the data set. If the name equals a pre-defined string use the customs settings.
     Otherwise, try to find the data set in the torchvision.datasets library. If not present, throw AttributeError.

     Parameters:
         path_for_data_set: String containing the path to the location of the data set.
         data_set_name: Name of the required data set.
     """
    logging.debug('Loading datasets..')
    # If True, use custom settings for this specific data set.
    if data_set_name == 'tiny-imagenet-100':
        transform_train = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.4802, 0.4481, 0.3975), (0.2296, 0.2263, 0.2255)),
        ])

        transform_test = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.4802, 0.4481, 0.3975), (0.2296, 0.2263, 0.2255)),
        ])

        train_set = datasets.ImageFolder(root=path_for_data_set + data_set_name + '/train',
                                         transform=transform_train
                                         )

        test_set = datasets.ImageFolder(root=path_for_data_set + data_set_name + '/test',
                                        transform=transform_test
                                        )

        logging.debug('Loading completed')

        return train_set, test_set

    else:
        if not hasattr(torchvision.datasets, data_set_name):
            # dataset not found exception
            error_message = f"...dataset \"{data_set_name}\" is not supported or cannot be found in TorchVision " \
                            f"Datasets! "
            logging.error(error_message)
            raise AttributeError(error_message)
        else:
            # prepare raw training & test datasets
            transform_train = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
            ])

            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
            ])

            training_data = torchvision.datasets.__dict__[data_set_name](
                root=path_for_data_set,
                train=True,
                download=True,
                transform=transform_train
            )

            test_data = torchvision.datasets.__dict__[data_set_name](
                root=path_for_data_set,
                train=False,
                download=True,
                transform=transform_test
            )
            logging.debug('Loading completed')

        return training_data, test_data


########################################################################################################################
# Classes for ConfusionMatrix and AverageMeter, used to track metrics during runtime.
########################################################################################################################

class ConfusionMatrix(object):
    """
    ConfusionMatrix Class
    Used to keep track of True/False positives and True/False negatives.

    Methods:
        reset: method to reset all attributes to zero.
        update_from_matrix: update all attributes with the scores from another ConfusionMatrix object.
        update: update all attributes by comparing the predictions and the labels.
        get_accuracy: calculate the accuracy based on the attributes.
        get_confusion_matrix: Getter method for the attributes of the Confusion Matrix.


     """

    def __init__(self):
        self.tp = 0
        self.fp = 0
        self.tn = 0
        self.fn = 0

    def reset(self):
        self.tp = 0
        self.fp = 0
        self.tn = 0
        self.fn = 0

    def update_from_matrix(self, confusion_matrix):
        (tp, fp, tn, fn) = confusion_matrix

        self.tp += tp
        self.fn += fn
        self.fp += fp
        self.tn += tn

    def update(self, predictions, labels):
        confusion_vector = (torch.round(predictions) / labels).float()

        self.tp += torch.sum(confusion_vector == 1.).item()
        self.fn += torch.sum(confusion_vector == 0.).item()
        self.fp += torch.sum(confusion_vector == float('inf')).item()
        self.tn += torch.sum(torch.isnan(confusion_vector)).item()

    def get_accuracy(self):
        return ((self.tp + self.tn) / (self.tp + self.tn + self.fp + self.fn)) * 100

    def get_confusion_matrix(self):
        return self.tp, self.fp, self.tn, self.fn


class AverageMeter(object):
    """
        Computes and stores the average and current value
        Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """

    def __init__(self):
        self.total = None
        self.sum = None
        self.avg = None
        self.correct = None
        self.reset()

    def reset(self):
        self.correct = 0
        self.avg = 0
        self.sum = 0
        self.total = 0

    def update(self, correct, n=1):
        self.correct = correct
        self.sum += correct * n
        self.total += n
        self.avg = self.sum / self.total
