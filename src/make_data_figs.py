from data import DataManager
from configuration import Config
import os

FIG_PATH = "../fig"

configuration_path = 'configurations' + os.sep + 'exp10.yaml'
config = Config(configuration_path)
data_manager = DataManager(config)