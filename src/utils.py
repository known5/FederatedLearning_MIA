import logging
import time
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.utils.data as data_util

from src.models import TestNet, AlexNet


def get_duration(start_time):
    hours, rem = divmod(time.time() - start_time, 3600)
    minutes, seconds = divmod(rem, 60)
    return "{:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds)


def load_model(model_name, is_local_model):
    """ Ja hier moet dus documentatie """
    if is_local_model:
        model = AlexNet(num_classes=200)
    else:
        if not hasattr(models, model_name):
            error_message = f"...model \"{model_name}\" is not supported or cannot be found in TorchVision models!"
            logging.error(error_message)
            raise AttributeError(error_message)
        else:
            model = models.__dict__[model_name]
    return model


def calculate_mean_std_(data_path, data_name):
    train_set = datasets.ImageFolder(data_path + data_name + '/train', transform=transforms.ToTensor())

    loader = data_util.DataLoader(train_set, batch_size=8)

    nimages = 0
    mean = 0.
    std = 0.
    for batch, _ in loader:
        # Rearrange batch to be the shape of [B, C, W * H]
        batch = batch.view(batch.size(0), batch.size(1), -1)
        # Update total number of images
        nimages += batch.size(0)
        # Compute mean and std here
        mean += batch.mean(2).sum(0)
        std += batch.std(2).sum(0)

    # Final step
    mean /= nimages
    std /= nimages

    print(mean)
    print(std)


def load_dataset(data_path, data_name, number_of_clients, is_iid):
    """ Ja hier moet dus documentatie """
    logging.debug('Loading datasets..')

    if data_name == 'tiny-imagenet-200':
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

        train_set = datasets.ImageFolder(root=data_path + data_name + '/train',
                                         transform=transform_train
                                         )
        val_set = datasets.ImageFolder(root=data_path + data_name + '/val',
                                       transform=transform_train
                                       )
        test_set = datasets.ImageFolder(root=data_path + data_name + '/test',
                                        transform=transform_test
                                        )

        logging.debug('Loading completed')
        return train_set, test_set, val_set

    else:

        if not hasattr(torchvision.datasets, data_name):
            # dataset not found exception
            error_message = f"...dataset \"{data_name}\" is not supported or cannot be found in TorchVision Datasets!"
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

            training_data = torchvision.datasets.__dict__[data_name](
                root=data_path,
                train=True,
                download=True,
                transform=transform_train
            )
            test_data = torchvision.datasets.__dict__[data_name](
                root=data_path,
                train=False,
                download=True,
                transform=transform_test
            )
            logging.debug('Loading completed')

        return training_data, test_data


class AverageMeter(object):
    """
        Computes and stores the average and current value
        Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
        Extended with the history list variable
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
        self.sum += correct
        self.total += n
        self.avg = self.sum / self.total
