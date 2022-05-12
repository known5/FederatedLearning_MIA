from .client import Client
from .models import *


class Attacker(Client):

    def __init__(self,
                 client_id,
                 local_data,
                 device,
                 target_train_model
                 ):
        """ Ja hier moet dus documentatie """
        super().__init__(client_id, local_data, device)
        self.exploit_gradient = None
        self.exploit_last_layer = None
        self.encoder = None
        self.output_size = None
        self.layers_to_exploit = None
        self.gradients_to_exploit = None
        self.exploit_loss = True,
        self.exploit_label = True,
        self.learning_rate = 0.001
        self.epochs = 1
        self.target_train_model = target_train_model

        self.create_attack_components()

    def create_attack_components(self):
        """ Ja hier moet dus documentatie """





