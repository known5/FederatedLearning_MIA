import torch.nn as nn
import torch.nn.functional as F

from src.utils import *


class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=96, kernel_size=28, stride=4, padding=0)
        self.maxpool = nn.MaxPool2d(kernel_size=1, stride=2)
        self.conv2 = nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, stride=1, padding=2)
        self.conv3 = nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(in_features=256, out_features=4096)
        self.fc2 = nn.Linear(in_features=4096, out_features=4096)
        self.fc3 = nn.Linear(in_features=4096, out_features=10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.maxpool(x)
        x = F.relu(self.conv2(x))
        x = self.maxpool(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = self.maxpool(x)
        x = x.reshape(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


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
