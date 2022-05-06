import torchaudio
import random
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

##############################################
# Dataset
##############################################
def speechcommands_file_reader(path):
    """
    
    """
    labels = ["backward", "bed", "bird", "cat", "dog", "down", "eight", "five", \
              "follow", "forward", "four", "go", "happy", "house", "learn", \
              "left", "marvin", "nine", "no", "off", "on", "one", "right", \
              "seven", "sheila", "six", "stop", "three", "tree", "two", "up", \
              "visual", "wow", "yes", "zero"]

    label2id = {}
    id2label = {}
    for i, label in enumerate(labels):
        label2id[label] = i
        id2label[i] = label

    flist = []
    for label in labels:
        sub_path = os.path.join(path, label)
        id = label2id[label]
        for f in os.listdir(sub_path):
            total_path = os.path.join(sub_path, f)
            flist.append((id, total_path))

    return label2id, id2label, flist

def train_test_split(flist, test_split):
    n = len(flist)
    random.shuffle(flist)
    return flist[: -int(test_split*n)],   flist[-int(test_split*n):]


class speechCommandDataset(Dataset):
    def __init__(
        self, 
        flist,
        config,
        audio_length=16000
        ):

        self.flist = flist
        self.config = config
        self.audio_length = audio_length
        self.loader = torchaudio.load

    def __len__(self):
        return len(self.flist)

    def __getitem__(self, index):
        label, f = self.flist[index]
        audio, samplerate = self.loader(f)

        residual = self.audio_length - audio.size(1)
        if residual:
            left_pad = np.random.choice(residual)
            right_pad = residual - left_pad
            p1d = (left_pad, right_pad) # pad last dim by num_pad on the right side
            audio = F.pad(audio, p1d, "constant", 0) 

        label_onehot = torch.zeros(self.config['num_classes'], dtype=torch.float32)
        #print(label_onehot)
        #print(label)
        label_onehot[int(label)] = 1

        return audio, label_onehot
    
    def getDataloader(self, batchSize):
        print("hi dataloader!")
        return DataLoader(
            dataset=self, 
            batch_size=batchSize,
            collate_fn = collate_fn_dense,
            shuffle=True,
            drop_last=True
        )

def collate_fn_dense(batch):
    """
    squeeze the data from (batch * 1 * audio_length) to (batch * 1 * audio_length) 
    only suitable for VAE with dense-layer
    """
    audios, targets = [], []

    for audio, label in batch:
        audios += [audio]
        targets += [label]

        #targets.append(int(label))
        #print(label)
        #print(label)

    #print(targets)
    audios = torch.vstack(audios)
    targets = torch.vstack(targets)
    #targets = np.array(targets)
    targets = torch.tensor(targets, dtype=torch.long)

    return audios, targets


def collate_fn_1dcc(batch):
    pass