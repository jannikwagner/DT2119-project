import torch
import dataclasses

@dataclasses.dataclass
class Config:
    DATA_DOWNLOAD_PATH = "../data/sc/"
    DATA_PATH = "../data/sc/SpeechCommands/speech_commands_v0.02"
    PICKLE_DICT = "../simple_pickles/"
    AUDIO_PATH = "../audio/"

    should_repickle = False
    should_train_model = False

    device = "cuda" if torch.cuda.is_available() else "cpu"

    epochs = 5
    batch_size = 256

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


config = Config()
