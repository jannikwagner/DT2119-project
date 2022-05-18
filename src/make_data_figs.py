import unittest
from data import DataManager
from configuration import Config
import os
import matplotlib.pyplot as plt

FIG_PATH = "../fig"

configuration_path = 'configurations' + os.sep + 'exp10.yaml'
config = Config(configuration_path)
data_manager = DataManager(config, print_info=False)
some_utterances = [0] #[0, 25, 123, 220]
for i in some_utterances:
    print()
    """ mel_spec = data_manager.transform(data_manager.train_set[i][0])[0]
    plt.pcolormesh(mel_spec)
    plt.show() """