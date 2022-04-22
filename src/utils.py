import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
import torch.optim as optimizers
import torch.nn as nn
from src.models import AlexNet


def init_weights_normal(layer):
    if isinstance(layer, nn.Linear):
        nn.init.normal(layer.weight, 0.0, 0.01)
        if layer.bias is not None:
            nn.init.zeros_(layer.bias)


def select_transformation(data_name):
    """ Ja hier moet dus documentatie """
    transform_train, transform_test = None, None
    if data_name in ["CIFAR100"]:
        transform_train = transforms.Compose([
            transforms.Resize(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

    if data_name in ["EMNIST"]:
        transform_train = transforms.Compose([
            transforms.ToTensor()
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor()
        ])
    return transform_train, transform_test


def load_model(model_name, is_local_model, is_pretrained):
    """ Ja hier moet dus documentatie """
    print("Loading models... ")
    model = None
    if is_local_model:
        model = AlexNet()
    else:
        if not hasattr(models, model_name):
            error_message = f"...model \"{model_name}\" is not supported or cannot be found in TorchVision models!"
            raise AttributeError(error_message)
        else:
            model = models.__dict__[model_name](is_pretrained)
    print(f"Successfully loaded model: \"{model_name}\"...")
    return model


def load_dataset(data_path, data_name, number_of_clients, is_iid):
    """ Ja hier moet dus documentatie """
    print("Loading datasets...")

    if not hasattr(torchvision.datasets, data_name):
        # dataset not found exception
        error_message = f"...dataset \"{data_name}\" is not supported or cannot be found in TorchVision Datasets!"
        raise AttributeError(error_message)
    else:
        # prepare raw training & test datasets
        transform_train, transform_test = select_transformation(data_name)
        training_data = torchvision.datasets.__dict__[data_name](
            root=data_path,
            # split='balanced',
            train=True,
            download=True,
            transform=transform_train
        )
        test_data = torchvision.datasets.__dict__[data_name](
            root=data_path,
            # split='balanced',
            train=False,
            download=True,
            transform=transform_test
        )

    return training_data, test_data
