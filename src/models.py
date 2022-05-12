import torch.nn as nn
import torch.nn.functional as F
import torch

from src.utils import *


class TestNetCNN(nn.Module):

    def __init__(self, input_channels=3, output_classes=100):
        super(TestNetCNN, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=input_channels,
                      out_channels=6,
                      kernel_size=3,
                      padding=1),
            nn.Conv2d(in_channels=6,
                      out_channels=9,
                      kernel_size=3,
                      padding=1)
        )
        self.flatten = nn.Flatten()
        self.classifier = nn.Sequential(
            nn.Linear(9216, 2048),
            nn.ReLU(),
            nn.Linear(2048, 256),
            nn.ReLU(),
            nn.Linear(256, output_classes)
        )

    def forward(self, x):
        x = self.cnn(x)
        x = self.flatten(x)
        return self.classifier(x)


class TestNet(nn.Module):

    def __init__(self, input_size=3 * 32 * 32, output_classes=100):
        super(TestNet, self).__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_size, output_classes),
            nn.ReLU()
        )

    def forward(self, x):
        return self.net(x)


class AlexNet(nn.Module):

    def __init__(self, classes=100):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
            # nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 1 * 1, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, classes),
        )

    def forward(self, x):
        x = torch.unsqueeze(x, dim=2)
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class AttackModel(nn.Module):
    """ Ja hier moet dus documentatie """

    def __int__(self, target_model):
        super().__init__()
        self.encoder_inputs = []

        # create component if last layer is to be exploited
        if self.exploit_last_layer:
            self.layer_component = FcnComponent(target_model.output, 100)
            self.encoder_inputs.append(self.layer_component.output)

            # create component if OHE label is to be exploited
        if self.exploit_label:
            self.label_component = FcnComponent(target_model.output)
            self.encoder_inputs.append(self.label_component.output)

            # create component if loss value is to be exploited
        if self.exploit_loss:
            self.loss_component = FcnComponent(1, 100)
            self.encoder_inputs.append(self.loss_component.output)

            # creates CNN/FCN component for gradient values of layers of gradients to exploit
        if self.exploit_gradient:
            self.gradient_component = CnnForFcnGradComponent(target_model[11].weight.grad)
            self.encoder_inputs.append(self.gradient_component.output)

        self.encoder = AutoEncoder(self.encoder_inputs)

    def forward(self, x1, x2, x3, x4):
        x1 = self.layer_component(x1)
        x2 = self.label_component(x2)
        x3 = self.loss_component(x3)
        x4 = self.gradient_component(x4)

        x = torch.cat((x1, x2, x3, x4), dim=1)
        self.classifier(x)
        return x


class AutoEncoder(nn.Module):
    """ Ja hier moet dus documentatie """

    def __init__(self, encoder_input):
        """ Ja hier moet dus documentatie """
        super(AutoEncoder, self).__init__()
        self.input = encoder_input
        self.output = 1
        self.encoder = nn.Sequential(
            nn.Linear(encoder_input, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        """ Ja hier moet dus documentatie """
        return self.encoder(x)


class FcnComponent(nn.Module):

    def __init__(self, input_size, layer_size=128):
        """ Ja hier moet dus documentatie """
        super(FcnComponent, self).__init__()
        self.input = input_size
        self.output = 64
        self.fcn = nn.Sequential(
            nn.Linear(input_size, layer_size),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        self.fcn.apply(init_weights_normal)

    def forward(self, x):
        """ Ja hier moet dus documentatie """
        return self.fcn(x)


class CnnForFcnGradComponent(nn.Module):

    def __init__(self, input_size):
        """ Ja hier moet dus documentatie """
        super(CnnForFcnGradComponent, self).__init__()
        dim = int(input_size[1])
        self.input = 0
        self.output = 256

        self.cnn = nn.Sequential(
            nn.Dropout(0.2),
            nn.Conv2d(in_channels=dim,
                      out_channels=100,
                      kernel_size=(1, dim),
                      stride=(1, 1),
                      padding='valid'),
            nn.ReLU(),
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(2024, 2024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, self.output),
            nn.ReLU()
        )
        self.cnn.apply(init_weights_normal)

    def forward(self, x):
        """ Ja hier moet dus documentatie """
        return self.cnn(x)


class CnnForCnnLayerOutputsComponent(nn.Module):

    def __init__(self, input_size):
        """ Ja hier moet dus documentatie """
        super(CnnForCnnLayerOutputsComponent, self).__init__()

        dim2 = int(input_size[1])
        dim4 = int(input_size[3])

        self.input = dim4
        self.output = 64

        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=dim2,
                      out_channels=dim4,
                      kernel_size=(1, 1),
                      padding='valid'
                      ),
            nn.ReLU(),
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(in_features=dim4,
                      out_features=1024,
                      ),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(in_features=1024,
                      out_features=512,
                      ),
            nn.ReLU(),
            nn.Linear(in_features=512,
                      out_features=128,
                      ),
            nn.ReLU(),
            nn.Linear(in_features=128,
                      out_features=64,
                      ),
            nn.ReLU(),
        )
        self.cnn.apply(init_weights_normal)

    def forward(self, x):
        """ Ja hier moet dus documentatie """
        return self.cnn(x)


class CnnForCnnGradComponent(nn.Module):

    def __init__(self, input_size):
        """ Ja hier moet dus documentatie """
        super(CnnForCnnGradComponent, self).__init__()

        dim1 = int(input_size[3])
        dim2 = int(input_size[0])
        dim3 = int(input_size[1])

        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=dim3,
                      out_channels=dim1,
                      kernel_size=(dim2, dim3),
                      stride=(1, 1),
                      padding='same',
                      ),
            nn.ReLU(),
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(in_features=dim1,
                      out_features=64
                      ),
            nn.ReLU()
        )

        self.cnn.apply(init_weights_normal)

    def forward(self, x):
        """ Ja hier moet dus documentatie """
        return self.cnn(x)
