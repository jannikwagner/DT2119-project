import os
import yaml

from error_handling.console_logger import ConsoleLogger
from models.convolutional_vq_vae import ConvolutionalVQVAE
from experiments.device_configuration import DeviceConfiguration

configuration_path = "configurations/example.json"
default_configuration_path = 'configurations' + os.sep + 'vctk_features.yaml'

def load_configuration(path):
    configuration = None
    with open(path, 'r') as configuration_file:
        configuration = yaml.load(configuration_file, Loader=yaml.FullLoader)
    return configuration

config = load_configuration(default_configuration_path)

print(config)

device_config = DeviceConfiguration.load_from_configuration(config)
model = ConvolutionalVQVAE(config, device_config.device).to(device_config.device)

print(model)