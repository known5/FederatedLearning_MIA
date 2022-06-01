import torch.nn as nn
import torch

""" Ja hier moet dus documentatie """


########################################################################################################################
# Model Class Helpers
########################################################################################################################


def init_weights_normal(layer):
    """
     Applies L2 regularization to a given model layer
        Set to use normal distribution
     """
    if isinstance(layer, nn.Linear):
        nn.init.normal_(layer.weight, 0.0, 0.01)
        if layer.bias is not None:
            nn.init.zeros_(layer.bias)


def get_output_shape_of_last_layer(target_model):
    """
        Gets the output shape of the last layer of a given model
    """
    output_shape = None
    for name, param in target_model.named_parameters():
        if 'weight' in name:
            output_shape = param.size()[1]
    return output_shape


def get_last_layer_name(target_model):
    """
         Gets the output shape of the last layer of a given model
    """
    layer_name = None
    for name, param in target_model.named_parameters():
        if 'weight' in name:
            layer_name = name
    return layer_name


########################################################################################################################
# Models
########################################################################################################################


class TestNet(nn.Module):

    def __init__(self, input_size=3 * 32 * 32, output_classes=100):
        super(TestNet, self).__init__()

        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_size, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, output_classes)
        )

    def forward(self, x):
        return self.net(x)


class AlexNet(nn.Module):

    def __init__(self, num_classes=10):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class AttackModel(nn.Module):
    """ Ja hier moet dus documentatie """

    def __init__(self, target_model,
                 exploit_last_layer=True,
                 exploit_label=True,
                 exploit_loss=True,
                 exploit_gradient=True
                 ):
        super(AttackModel, self).__init__()
        self.target_output = get_output_shape_of_last_layer(target_model)
        self.last_layer_name = get_last_layer_name(target_model)

        self.encoder_inputs = []
        self.model = target_model

        self.exploit_last_layer = exploit_last_layer
        self.exploit_label = exploit_label
        self.exploit_loss = exploit_loss
        self.exploit_gradient = exploit_gradient

        self.layer_component = None
        self.label_component = None
        self.loss_component = None
        self.gradient_component = None

        # create component if last layer is to be exploited
        if exploit_last_layer:
            self.layer_component = FcnComponent(self.target_output, 100)
            module_output = get_output_shape_of_last_layer(self.layer_component)
            self.encoder_inputs.append(module_output)

            # create component if OHE label is to be exploited
        if exploit_label:
            self.label_component = FcnComponent(self.target_output)
            module_output = get_output_shape_of_last_layer(self.label_component)
            self.encoder_inputs.append(module_output)

            # create component if loss value is to be exploited
        if exploit_loss:
            self.loss_component = FcnComponent(1, 100)
            module_output = get_output_shape_of_last_layer(self.loss_component)
            self.encoder_inputs.append(module_output)

            # creates CNN/FCN component for gradient values of layers of gradients to exploit
        if exploit_gradient:
            self.gradient_component = CnnForFcnGradComponent(self.target_output)
            module_output = get_output_shape_of_last_layer(self.gradient_component)
            self.encoder_inputs.append(module_output)

        self.encoder = AutoEncoder(self.encoder_inputs)

    def forward(self, layer=torch.Tensor, labels=torch.Tensor, loss=torch.Tensor,
                gradient=torch.Tensor) -> torch.Tensor:
        inputs = []
        if self.exploit_last_layer:
            inputs.append(self.layer_component(layer))

        if self.exploit_label:
            inputs.append(self.label_component(labels))

        if self.exploit_loss:
            inputs.append(self.loss_component(loss))

        if self.exploit_gradient:
            inputs.append(self.gradient_component(gradient))

        x = torch.cat(inputs, dim=1)
        self.classifier(x)
        return x


class AutoEncoder(nn.Module):
    """ Ja hier moet dus documentatie """

    def __init__(self, encoder_input):
        """ Ja hier moet dus documentatie """
        super(AutoEncoder, self).__init__()
        input_concat = sum(encoder_input)

        self.encoder = nn.Sequential(
            nn.Linear(input_concat, 256),
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
        dim = input_size

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
