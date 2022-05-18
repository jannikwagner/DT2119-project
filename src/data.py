import torch
import os
import pickle
import torch
import torchaudio
from torchaudio.datasets import SPEECHCOMMANDS
import torch.nn as nn

from configuration import Config



##############################################
# LOADING THE DATA
###############################################
class SubsetSC(SPEECHCOMMANDS):
    def __init__(self, path, subset: str = None, num: int = -1):
        super().__init__(path, download=True)

        def load_list(filename):
            filepath = os.path.join(self._path, filename)
            with open(filepath) as fileobj:
                return [os.path.normpath(os.path.join(self._path, line.strip())) for line in fileobj]

        if subset == "validation":
            self._walker = load_list("validation_list.txt")
        elif subset == "testing":
            self._walker = load_list("testing_list.txt")
        elif subset == "training":
            excludes = load_list("validation_list.txt") + load_list("testing_list.txt")
            excludes = set(excludes)
            self._walker = [w for w in self._walker if w not in excludes]
            self._walker = self._walker[:num]

class DataManager:  # needs modularization!
    def __init__(self, config: Config):
        # Create training and testing split of the data. We do not use validation.
        self.train_set = SubsetSC(config.DATA_DOWNLOAD_PATH, "training", config.dummy_data_length)
        print("trainset length", len(self.train_set))
        self.test_set = SubsetSC(config.DATA_DOWNLOAD_PATH, "testing")
        print("SubsetSC loaded")

        waveform, sample_rate, label, speaker_id, utterance_number = self.train_set[0]
        self.sample_rate = sample_rate

        if config.should_repickle:
            self.labels = sorted(list(set(datapoint[2] for datapoint in self.train_set)))
            with open(config.LABELS_PATH, 'wb') as handle:
                pickle.dump(self.labels, handle, protocol=pickle.HIGHEST_PROTOCOL)
            print("labels saved")  
        else:
            with open(config.LABELS_PATH, 'rb') as handle:
                self.labels = pickle.load(handle)
            print("labels loaded")
        print(self.labels)
        print("num labels", len(self.labels))
        self.n_labels = len(self.labels)
        self.label_dic = {label: i for i, label in enumerate(self.labels)}

        def make_speaker_dic(data_set):
            speakers = [speaker_id for _, _, _, speaker_id, _ in data_set]
            speakers = set(speakers)
            speakers = sorted([speaker for speaker in speakers])
            speaker_dic = {speaker: i for i, speaker in enumerate(speakers)}
            return speaker_dic

        if config.should_repickle: 
            self.speaker_dic = make_speaker_dic(self.train_set)
            with open(config.SPEAKER_DICT_PATH, 'wb') as handle:
                pickle.dump(self.speaker_dic, handle, protocol=pickle.HIGHEST_PROTOCOL)
            print("speaker_dic saved")  
        else:
            with open(config.SPEAKER_DICT_PATH, 'rb') as handle:
                self.speaker_dic = pickle.load(handle)
            print("speaker dictionary loaded")
        self.speakers = list(self.speaker_dic)
        print("num spekaers", len(self.speakers))
        self.n_speakers = len(self.speakers)

        ##############################################
        # FORMATTING THE DATA
        ###############################################
        # how to use librosa mel filter defaults?
        self.transform_SpectrogramComplex = torchaudio.transforms.Spectrogram(n_fft=config.n_fft, win_length=config.win_length, hop_length=config.hop_length, power=None)  # hard to use in nn
        self.transform_Spectrogram = torchaudio.transforms.Spectrogram(n_fft=config.n_fft, win_length=config.win_length, hop_length=config.hop_length)
        self.transform_InverseSpectrogram = torchaudio.transforms.InverseSpectrogram(n_fft=config.n_fft, win_length=config.win_length, hop_length=config.hop_length)  # need complex
        self.transform_GriffinLim = torchaudio.transforms.GriffinLim(n_fft=config.n_fft, hop_length=config.hop_length, win_length=config.win_length)  # does not sound too good

        self.transform_MelScale = torchaudio.transforms.MelScale(n_mels=config.n_mels, sample_rate=sample_rate, mel_scale=config.mel_scale, n_stft=config.n_stft)
        self.transform_InverseMelScale = torchaudio.transforms.InverseMelScale(n_mels=config.n_mels, sample_rate=sample_rate, mel_scale=config.mel_scale, n_stft=config.n_stft)  # takes some time
        self.transform_MelSpectrogram = torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate, n_mels=config.n_mels, mel_scale=config.mel_scale, n_fft=config.n_fft, hop_length=config.hop_length, win_length=config.win_length)

        if config.transform_type == "spectrogram":
            self.transform = self.transform_Spectrogram
            self.inverse_transform = self.transform_GriffinLim
        elif config.transform_type == "mel_spectrogram":
            self.transform = self.transform_MelSpectrogram
            self.inverse_transform = nn.Sequential(self.transform_InverseMelScale, self.transform_GriffinLim)
        def tests(train_set):
            waveform, sample_rate, label, speaker_id, utterance_number = train_set[0]
            print(waveform, sample_rate, label, speaker_id, utterance_number)
            print(waveform.min(), waveform.max())
            spectrogram = self.transform_Spectrogram(waveform)
            print("spectrogram shape:", spectrogram.shape)
            spectrogram_complex = self.transform_SpectrogramComplex(waveform)
            print("spectrogram_complex shape:", spectrogram_complex.shape)
            mel_spectrogram = self.transform_MelSpectrogram(waveform)
            print("mel_spectrogram shape:", mel_spectrogram.shape)
            mel_spectrogram2 = self.transform_MelScale(spectrogram)
            print("mel_spectrogram2 shape:", mel_spectrogram2.shape)
            reconstructed_spectrogram = self.transform_InverseMelScale(mel_spectrogram)
            print("reconstructed_spectrogram shape:", reconstructed_spectrogram.shape)
            reconstructed_spectrogram2 = self.transform_InverseMelScale(mel_spectrogram2)
            print("reconstructed_spectrogram2 shape:", reconstructed_spectrogram2.shape)
            reconstructed = self.transform_GriffinLim(spectrogram)
            print("reconstructed shape:", reconstructed.shape)
            reconstructed2 = self.transform_InverseSpectrogram(spectrogram_complex)
            print("reconstructed2 shape:", reconstructed2.shape)
            reconstructed3 = self.transform_GriffinLim(reconstructed_spectrogram)
            print("reconstructed3 shape:", reconstructed3.shape)
            torchaudio.save(config.AUDIO_PATH+"reconstructed.wav", reconstructed, sample_rate)  # sounds bad
            torchaudio.save(config.AUDIO_PATH+"reconstructed2.wav", reconstructed2, sample_rate)  # sounds good
            torchaudio.save(config.AUDIO_PATH+"reconstructed3.wav", reconstructed3, sample_rate)  # sounds bad
            torchaudio.save(config.AUDIO_PATH+"original.wav", waveform, sample_rate)
            print(reconstructed-reconstructed2)
            print(reconstructed-reconstructed3)

        def get_data_dim(train_set, transform):
            waveform, sample_rate, label, speaker_id, utterance_number = train_set[0]
            transformed = transform(waveform)
            data_dim = transformed.shape
            return data_dim
        self.data_dim = get_data_dim(self.train_set, self.transform)
        # new_sample_rate = 8000
        # transform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=new_sample_rate)

        def pad_sequence(batch):
            # Make all tensor in a batch the same length by padding with zeros
            batch = [item.t() for item in batch]
            batch = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=0.)
            return batch.permute(0, 2, 1)
        
        def get_one_hot(val, length):
            arr = torch.zeros(length)
            arr[val] = 1
            # print(arr)
            return arr

        def collate_fn(batch):

            # A data tuple has the form:
            # waveform, sample_rate, label, speaker_id, utterance_number

            tensors, labels, speaker_ids = [], [], []

            # Gather in lists, and encode labels as indices
            for waveform, _, label, speaker_id,*_ in batch:
                tensors += [waveform]
                # print(label)
                label_idx = self.label_dic[label]
                # print(label_idx)
                label_idx = get_one_hot(label_idx, len(self.labels))
                # print(label_idx)
                speaker_idx = self.speaker_dic[speaker_id]
                speaker_idx = get_one_hot(speaker_idx, len(self.speakers))
                labels += [label_idx]
                speaker_ids += [speaker_idx]

            # Group the list of tensors into a batched tensor
            tensors = pad_sequence(tensors)
            labels = torch.stack(labels)
            speaker_ids = torch.stack(speaker_ids)

            return tensors, labels, speaker_ids

        if config.device == "cuda":
            num_workers = 1
            pin_memory = True
        else:
            num_workers = 0
            pin_memory = False

        self.train_loader = torch.utils.data.DataLoader(
            self.train_set,
            batch_size=config.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        self.test_loader = torch.utils.data.DataLoader(
            self.test_set,
            batch_size=config.batch_size,
            shuffle=False,
            drop_last=False,
            collate_fn=collate_fn,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
