import torch
import dataclasses
import os
import yaml

from typing import Any

"""
The goal will be to dynamically load configurations from yaml files. Until then, this will serve as interface.
"""

@dataclasses.dataclass
class Config:
    # DATA_DOWNLOAD_PATH = "../data/sc/"
    # DATA_PATH = "../data/sc/SpeechCommands/speech_commands_v0.02"
    # PICKLE_PATH = "../simple_pickles/"
    # AUDIO_PATH = "../audio/"
    # EXPERIMENTS_PATH = "../experiments/"

    # should_repickle = False
    # should_train_model = True


    # log_interval = 10

    # epochs = 2
    # batch_size = 256

    # model_type = "linear_vae"
    # latent_dim = 256
    # hidden_dims = [1024, 512, 512]

    # experiment_name = "exp3"

    # n_mels = 80
    # mel_scale = "htk"
    # n_fft = 400
    # hop_length = 256
    # win_length = 400
    # n_stft = n_fft // 2 + 1

    # n_mels = 39
    # mel_scale = "htk"
    # n_fft = 400
    # hop_length = None
    # win_length = None
    # n_stft = n_fft // 2 + 1

    # EXPERIMENT_PATH = os.path.join(EXPERIMENTS_PATH, experiment_name)
    # TRAINED_MODEL_PATH = os.path.join(EXPERIMENT_PATH,'trained_simple_vae_model_' + str(epochs) + 'epochs.pth')
    # LABELS_PATH = os.path.join(PICKLE_PATH, 'labels.pickle')
    # SPEAKER_DICT_PATH = os.path.join(PICKLE_PATH, 'speaker_dict.pickle')


    def load_configuration(self, path):
        configuration = None
        with open(path, 'r') as configuration_file:
            configuration = yaml.load(configuration_file, Loader=yaml.FullLoader)
        return configuration

    def __init__(self, configuration_path):
        self.config = self.load_configuration(configuration_path)

        for k, v in self.config.items():
            if v == "None":
                v = None
            self.__setattr__(k, v)
        
        self.device = "cuda" if (self.device == "cuda" and torch.cuda.is_available()) else "cpu"
        self.n_stft = self.n_fft // 2 + 1


        self.EXPERIMENT_PATH = os.path.join(self.EXPERIMENTS_PATH, self.experiment_name)
        self.TRAINED_MODEL_PATH = os.path.join(self.EXPERIMENT_PATH,'trained_simple_vae_model_' + str(self.epochs) + 'epochs.pth')
        self.LABELS_PATH = os.path.join(self.PICKLE_PATH, 'labels.pickle')
        self.SPEAKER_DICT_PATH = os.path.join(self.PICKLE_PATH, 'speaker_dict.pickle')
        self.CONFIG_PATH = os.path.join(self.EXPERIMENT_PATH, 'config.yaml')

        os.makedirs(self.AUDIO_PATH, exist_ok=True)
        os.makedirs(self.EXPERIMENT_PATH, exist_ok=True)
        os.makedirs(self.PICKLE_PATH, exist_ok=True)        

        with open(self.CONFIG_PATH, 'w') as configuration_file:
            yaml.dump(self.config, configuration_file)


if __name__ == "__main__":
    pass
    
