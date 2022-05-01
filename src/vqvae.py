import yaml

from error_handling.console_logger import ConsoleLogger
from models.convolutional_vq_vae import ConvolutionalVQVAE

configuration_path = "configurations/example.json"

def load_configuration():
    configuration = None
    with open(configuration_path, 'r') as configuration_file:
        configuration = yaml.load(configuration_file, Loader=yaml.FullLoader)
    return configuration

print(load_configuration())