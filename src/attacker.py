from .client import Client


class Attacker(Client):

    def __init__(self, client_id, local_data, device):
        """ Ja hier moet dus documentatie """
        super().__init__(client_id, local_data, device)

    def load_attack_test_data(self):
        pass

    def load_attack_train_data(self):
        pass

    def load_attack_model(self):
        """ Ja hier moet dus documentatie """
        pass

    def train_attack_model(self):
        """ Ja hier moet dus documentatie """
        pass
