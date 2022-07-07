import torch
import torch.nn as nn

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
            output_shape = param.size()[0]
    return output_shape


def get_last_layer_name(target_model):
    """
         Gets the name of the last layer of a given model
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

    def __init__(self, num_classes=100):
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
                 number_of_observed_models=1,
                 number_of_classes=100
                 ):
        super(AttackModel, self).__init__()
        self.last_layer_name = get_last_layer_name(target_model)
        self.encoder_inputs = []
        self.model = target_model
        self.number_of_observed_models = number_of_observed_models

        # create component if last layer is to be exploited
        self.layer_component = FcnComponent(number_of_classes, 100)
        module_output = get_output_shape_of_last_layer(self.layer_component)
        self.encoder_inputs.append(module_output)

        # create component if OHE label is to be exploited
        self.label_component = FcnComponent(number_of_classes, 128)
        module_output = get_output_shape_of_last_layer(self.label_component)
        self.encoder_inputs.append(module_output)

        # create component if loss value is to be exploited
        self.loss_component = FcnComponent(1, number_of_classes)
        module_output = get_output_shape_of_last_layer(self.loss_component)
        self.encoder_inputs.append(module_output)

        # creates CNN/FCN component for gradient values of layers of gradients to exploit
        self.gradient_component = GradientComponent(number_of_classes)
        module_output = get_output_shape_of_last_layer(self.gradient_component)
        self.encoder_inputs.append(module_output)

        self.encoder = AutoEncoder(self.encoder_inputs, number_of_observed_models)
        self.output = nn.Sigmoid()

    def forward(self, model_predictions, encoded_labels, loss_values, gradients) -> torch.Tensor:

        for model_input in range(self.number_of_observed_models):
            temp = self.layer_component(model_predictions[model_input])
            temp1 = self.label_component(encoded_labels[model_input])
            temp2 = self.loss_component(loss_values[model_input])
            temp3 = self.gradient_component(gradients[model_input])

            temp1 = torch.unsqueeze(temp1, -1)
            if model_input == 0:
                x = torch.cat((temp, temp1, temp2, temp3), dim=1)
            else:
                x = torch.cat((temp, temp1, temp2, temp3, x), dim=1)

        is_member = self.encoder(x)
        return self.output(is_member)


class AutoEncoder(nn.Module):
    """ Ja hier moet dus documentatie """

    def __init__(self, encoder_input, number_of_observed_models):
        """ Ja hier moet dus documentatie """
        super(AutoEncoder, self).__init__()
        input_concat = sum(encoder_input)

        self.encoder = nn.Sequential(
            nn.Linear(input_concat * number_of_observed_models, 256),
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
            nn.Linear(layer_size, 64),
            nn.ReLU()
        )
        self.fcn.apply(init_weights_normal)

    def forward(self, x):
        """ Ja hier moet dus documentatie """
        return self.fcn(x)


class GradientComponent(nn.Module):

    def __init__(self, number_of_classes=100):
        """ Ja hier moet dus documentatie """
        super(GradientComponent, self).__init__()

        self.cnn = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Conv2d(1, 1000, kernel_size=(1, 100), stride=1),
            nn.ReLU(),
        )
        self.linear = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(256 * 1000, 1024),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
        )
        self.linear.apply(init_weights_normal)

    def forward(self, x):
        """ Ja hier moet dus documentatie """
        temp = self.cnn(x).view(x.size()[0], -1)
        return self.linear(temp)
