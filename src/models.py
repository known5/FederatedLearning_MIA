import torch.nn as nn
import torch.nn.functional as F
import torch

from src.utils import *


class TestNet(nn.Module):

    def __init__(self, input_size=3*32*32, output_classes=100):
        super(TestNet, self).__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_size, 1024),
            nn.ReLU(),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, output_classes)
        )

    def forward(self, x):
        return self.net(x)


class AutoEncoder(nn.Module):
    """ Ja hier moet dus documentatie """

    def __init__(self, encoder_input):
        """ Ja hier moet dus documentatie """
        super(AutoEncoder, self).__init__()
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
            nn.Linear(512, 256),
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
