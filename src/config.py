import torch
import dataclasses
import os

"""
The goal will be to dynamically load configurations from yaml files. Until then, this will serve as interface.
"""

@dataclasses.dataclass
class Config:
    DATA_DOWNLOAD_PATH = "../data/sc/"
    DATA_PATH = "../data/sc/SpeechCommands/speech_commands_v0.02"
    PICKLE_PATH = "../simple_pickles/"
    AUDIO_PATH = "../audio/"
    EXPERIMENTS_PATH = "../experiments/"

    should_repickle = False
    should_train_model = True

    device = "cuda" if torch.cuda.is_available() else "cpu"

    epochs = 2
    batch_size = 256
    latent_dim = 256

    experiment_name = "exp2"

    # n_mels = 80
    # mel_scale = "htk"
    # n_fft = 400
    # hop_length = 256
    # win_length = 400
    # n_stft = n_fft // 2 + 1

    n_mels = 39
    mel_scale = "htk"
    n_fft = 400
    hop_length = None
    win_length = None
    n_stft = n_fft // 2 + 1

    EXPERIMENT_PATH = os.path.join(EXPERIMENTS_PATH, experiment_name)
    TRAINED_MODEL_PATH = os.path.join(EXPERIMENT_PATH,'trained_simple_vae_model_' + str(epochs) + 'epochs.pth')


config = Config()

os.makedirs(config.DATA_DOWNLOAD_PATH, exist_ok=True)
os.makedirs(config.AUDIO_PATH, exist_ok=True)
os.makedirs(config.EXPERIMENT_PATH, exist_ok=True)
