import unittest
import wave
from data import DataManager
from configuration import Config
import os
import matplotlib.pyplot as plt
import numpy as np

import librosa
import librosa.display

FIG_PATH = "../fig/"

configuration_path = 'configurations' + os.sep + 'exp_draw.yaml'
config = Config(configuration_path)
data_manager = DataManager(config, print_info=False)
some_utterances = [0, 5000, 10000, 20000, 40000]
sr = 16000
for i in some_utterances:
    waveform, sample_rate, label, speaker_id, utterance_number = data_manager.train_set[i]
    waveform = np.array(waveform)
    librosa.display.waveshow(waveform, sr=sr)
    plt.title("Waveform for utterance " + label)
    plt.savefig(FIG_PATH+label+'_'+speaker_id+'_waveform.png')
    plt.cla()
    S = librosa.feature.melspectrogram(waveform, sr=sr, n_fft=config.n_fft, hop_length=config.hop_length, n_mels=config.n_mels)
    S_DB = librosa.power_to_db(S, ref=np.max)[0]
    librosa.display.specshow(S_DB, sr=sr, hop_length=256, x_axis='time', y_axis='mel')
    cb = plt.colorbar(format='%+2.0f dB')
    plt.title("Mel Spectogram for utterance " + label)
    plt.savefig(FIG_PATH+label+'_'+speaker_id+'_melspec.png')
    cb.remove()
    plt.cla()