import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
import torch.optim as optimizers
import torch.nn as nn
import src.models as ms


def init_weights_normal(layer):
    if isinstance(layer, nn.Linear):
        nn.init.normal(layer.weight, 0.0, 0.01)
        if layer.bias is not None:
            nn.init.zeros_(layer.bias)


def load_model(model_name, is_local_model, is_pretrained):
    """ Ja hier moet dus documentatie """
    model = None
    if is_local_model:
        model = ms.AlexNet()
    else:
        if not hasattr(models, model_name):
            error_message = f"...model \"{model_name}\" is not supported or cannot be found in TorchVision models!"
            raise AttributeError(error_message)
        else:
            model = models.__dict__[model_name](is_pretrained)
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

    return training_data, test_data
