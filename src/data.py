import torch
import os
import pickle
import torch
import torchaudio
from torchaudio.datasets import SPEECHCOMMANDS

from config import config

os.makedirs(config.DATA_DOWNLOAD_PATH, exist_ok=True)
os.makedirs(config.AUDIO_PATH, exist_ok=True)

##############################################
# LOADING THE DATA
###############################################
class SubsetSC(SPEECHCOMMANDS):
    def __init__(self, subset: str = None, num: int = -1):
        super().__init__(config.DATA_DOWNLOAD_PATH, download=True)

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


# Create training and testing split of the data. We do not use validation.
train_set = SubsetSC("training")
print("trainset length", len(train_set))
test_set = SubsetSC("testing")
print("SubsetSC loaded")

waveform, sample_rate, label, speaker_id, utterance_number = train_set[0]
print("waveforms loaded")

if config.should_repickle:
    labels = sorted(list(set(datapoint[2] for datapoint in train_set)))
    with open(config.PICKLE_DICT+'labels.pickle', 'wb') as handle:
        pickle.dump(labels, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print("labels saved")  
else:
    with open(config.PICKLE_DICT+'labels.pickle', 'rb') as handle:
        labels = pickle.load(handle)
    print("labels loaded")

def make_speaker_dic(data_set):
    speakers = [speaker_id for _, _, _, speaker_id, _ in data_set]
    speakers = set(speakers)
    speakers = sorted([speaker for speaker in speakers])
    speaker_dic = {speaker: i for i, speaker in enumerate(speakers)}
    return speaker_dic

if config.should_repickle: 
    speaker_dic = make_speaker_dic(train_set)
    with open(config.PICKLE_DICT+'speaker_dict.pickle', 'wb') as handle:
        pickle.dump(speaker_dic, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print("speaker_dic saved")  
else:
    with open(config.PICKLE_DICT+'speaker_dict.pickle', 'rb') as handle:
        speaker_dic = pickle.load(handle)
    print("speaker dictionary loaded")

##############################################
# FORMATTING THE DATA
###############################################
# how to use librosa mel filter defaults?
transform_SpectrogramComplex = torchaudio.transforms.Spectrogram(n_fft=config.n_fft, win_length=config.win_length, hop_length=config.hop_length, power=None)  # hard to use in nn
transform_Spectrogram = torchaudio.transforms.Spectrogram(n_fft=config.n_fft, win_length=config.win_length, hop_length=config.hop_length)
transform_InverseSpectrogram = torchaudio.transforms.InverseSpectrogram(n_fft=config.n_fft, win_length=config.win_length, hop_length=config.hop_length)  # need complex
transform_GriffinLim = torchaudio.transforms.GriffinLim(n_fft=config.n_fft, hop_length=config.hop_length, win_length=config.win_length)  # does not sound too good

transform_MelScale = torchaudio.transforms.MelScale(n_mels=config.n_mels, sample_rate=sample_rate, mel_scale=config.mel_scale, n_stft=config.n_stft)
transform_InverseMelScale = torchaudio.transforms.InverseMelScale(n_mels=config.n_mels, sample_rate=sample_rate, mel_scale=config.mel_scale, n_stft=config.n_stft)  # takes some time
transform_MelSpectrogram = torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate, n_mels=config.n_mels, mel_scale=config.mel_scale, n_fft=config.n_fft, hop_length=config.hop_length, win_length=config.win_length)

def tests(train_set):
    waveform, sample_rate, label, speaker_id, utterance_number = train_set[0]
    print(waveform, sample_rate, label, speaker_id, utterance_number)
    print(waveform.min(), waveform.max())
    spectrogram = transform_Spectrogram(waveform)
    print("spectrogram shape:", spectrogram.shape)
    spectrogram_complex = transform_SpectrogramComplex(waveform)
    print("spectrogram_complex shape:", spectrogram_complex.shape)
    mel_spectrogram = transform_MelSpectrogram(waveform)
    print("mel_spectrogram shape:", mel_spectrogram.shape)
    mel_spectrogram2 = transform_MelScale(spectrogram)
    print("mel_spectrogram2 shape:", mel_spectrogram2.shape)
    reconstructed_spectrogram = transform_InverseMelScale(mel_spectrogram)
    print("reconstructed_spectrogram shape:", reconstructed_spectrogram.shape)
    reconstructed_spectrogram2 = transform_InverseMelScale(mel_spectrogram2)
    print("reconstructed_spectrogram2 shape:", reconstructed_spectrogram2.shape)
    reconstructed = transform_GriffinLim(spectrogram)
    print("reconstructed shape:", reconstructed.shape)
    reconstructed2 = transform_InverseSpectrogram(spectrogram_complex)
    print("reconstructed2 shape:", reconstructed2.shape)
    reconstructed3 = transform_GriffinLim(reconstructed_spectrogram)
    print("reconstructed3 shape:", reconstructed3.shape)
    torchaudio.save(config.AUDIO_PATH+"reconstructed.wav", reconstructed, sample_rate)  # sounds bad
    torchaudio.save(config.AUDIO_PATH+"reconstructed2.wav", reconstructed2, sample_rate)  # sounds good
    torchaudio.save(config.AUDIO_PATH+"reconstructed3.wav", reconstructed3, sample_rate)  # sounds bad
    torchaudio.save(config.AUDIO_PATH+"original.wav", waveform, sample_rate)
    print(reconstructed-reconstructed2)
    print(reconstructed-reconstructed3)

transform = transform_MelSpectrogram
def get_data_dim(train_set):
    waveform, sample_rate, label, speaker_id, utterance_number = train_set[0]
    transformed = transform(waveform)
    data_dim = transformed.shape
    return data_dim
data_dim = get_data_dim(train_set)
# new_sample_rate = 8000
# transform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=new_sample_rate)


def label_to_index(word):
    # Return the position of the word in labels
    return torch.tensor(labels.index(word))


def index_to_label(index):
    # Return the word corresponding to the index in labels
    # This is the inverse of label_to_index
    print("index", index)
    return labels[index]

def pad_sequence(batch):
    # Make all tensor in a batch the same length by padding with zeros
    batch = [item.t() for item in batch]
    batch = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=0.)
    return batch.permute(0, 2, 1)


def collate_fn(batch):

    # A data tuple has the form:
    # waveform, sample_rate, label, speaker_id, utterance_number

    tensors, targets, speaker_ids = [], [], []

    # Gather in lists, and encode labels as indices
    for waveform, _, label, speaker_id,*_ in batch:
        tensors += [waveform]
        targets += [label_to_index(label)]
        speaker_ids += [speaker_dic[speaker_id]]

    # Group the list of tensors into a batched tensor
    tensors = pad_sequence(tensors)
    targets = torch.stack(targets)
    speaker_ids = torch.Tensor(speaker_ids)[:, None]

    return tensors, targets, speaker_ids

if config.device == "cuda":
    num_workers = 1
    pin_memory = True
else:
    num_workers = 0
    pin_memory = False

train_loader = torch.utils.data.DataLoader(
    train_set,
    batch_size=config.batch_size,
    shuffle=True,
    collate_fn=collate_fn,
    num_workers=num_workers,
    pin_memory=pin_memory,
)
test_loader = torch.utils.data.DataLoader(
    test_set,
    batch_size=config.batch_size,
    shuffle=False,
    drop_last=False,
    collate_fn=collate_fn,
    num_workers=num_workers,
    pin_memory=pin_memory,
)
