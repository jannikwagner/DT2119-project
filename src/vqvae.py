from distutils.command.config import config
import os
import yaml
import pickle
import torch
import matplotlib as plt
from tqdm import tqdm
from error_handling.console_logger import ConsoleLogger
from models.convolutional_vq_vae import ConvolutionalVQVAE
from experiments.device_configuration import DeviceConfiguration
from torchaudio.datasets import SPEECHCOMMANDS
import torchaudio

DATA_DOWNLOAD_PATH = "../data/sc/"
DATA_PATH = "../data/sc/SpeechCommands/speech_commands_v0.02"
PICKLE_DICT = "../pickles/"

os.makedirs(DATA_DOWNLOAD_PATH, exist_ok=True)

default_configuration_path = 'configurations' + os.sep + 'vctk_features.yaml'

def load_configuration(path):
    configuration = None
    with open(path, 'r') as configuration_file:
        configuration = yaml.load(configuration_file, Loader=yaml.FullLoader)
    return configuration

CONFIG = load_configuration(default_configuration_path)
#print(CONFIG)

##############################################
# CREATE MODEL
###############################################
device_config = DeviceConfiguration.load_from_configuration(CONFIG)
#model = ConvolutionalVQVAE(CONFIG, device_config.device).to(device_config.device)

#print(model)
MODEL_PATH = PICKLE_DICT+'vq_vae_model.pickle'
#torch.save(model, MODEL_PATH)
model = torch.load(MODEL_PATH)
print("model loaded")

##############################################
# LOADING THE DATA
###############################################
class SubsetSC(SPEECHCOMMANDS):
    def __init__(self, subset: str = None):
        super().__init__(DATA_DOWNLOAD_PATH, download=True)

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


# Create training and testing split of the data. We do not use validation in this tutorial.
train_set = SubsetSC("training")
test_set = SubsetSC("testing")
print("SubsetSC loaded")

waveform, sample_rate, label, speaker_id, utterance_number = train_set[0]
print("waveforms loaded")

labels = sorted(list(set(datapoint[2] for datapoint in train_set)))
print("labels loaded")

def make_speaker_dic(data_set):
    speakers = [speaker_id for _, _, _, speaker_id, _ in data_set]
    speakers = set(speakers)
    speakers = sorted([speaker for speaker in speakers])
    speaker_dic = {speaker: i for i, speaker in enumerate(speakers)}
    return speaker_dic

# speaker_dic = make_speaker_dic(train_set)
# with open(PICKLE_DICT+'speaker_dict.pickle', 'wb') as handle:
#     pickle.dump(speaker_dic, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open(PICKLE_DICT+'speaker_dict.pickle', 'rb') as handle:
    speaker_dic = pickle.load(handle)
print("speaker dictionary loaded")


##############################################
# FORMATTING THE DATA
###############################################
n_mels=64
transform = torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_mels=39)

def label_to_index(word):
    # Return the position of the word in labels
    return torch.tensor(labels.index(word))


def index_to_label(index):
    # Return the word corresponding to the index in labels
    # This is the inverse of label_to_index
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


batch_size = 256

if device_config.device == "cuda":
    num_workers = 1
    pin_memory = True
else:
    num_workers = 0
    pin_memory = False

train_loader = torch.utils.data.DataLoader(
    train_set,
    batch_size=batch_size,
    shuffle=True,
    collate_fn=collate_fn,
    num_workers=num_workers,
    pin_memory=pin_memory,
)
test_loader = torch.utils.data.DataLoader(
    test_set,
    batch_size=batch_size,
    shuffle=False,
    drop_last=False,
    collate_fn=collate_fn,
    num_workers=num_workers,
    pin_memory=pin_memory,
)




##############################################
# TRAINING
###############################################

optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0.0001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)  # reduce the learning after 20 epochs by a factor of 10

def train(model, epoch, log_interval):
    model.train()
    for batch_idx, (data, target, speaker_id) in enumerate(train_loader):
        device = device_config.device
        data = data.to(device)
        target = target.to(device)
        speaker_id = target.to(speaker_id)
        print(data.shape, target.shape, speaker_id.shape)

        # apply transform and model on whole batch directly on device
        data = transform(data)
        print(data.shape)
        data = data[:, 0, ...]
        print(data.shape)
        output = model(data, speaker_dic, speaker_id)
        reconstructed_x, vq_loss, losses, perplexity, encoding_indices, concatenated_quantized = output
        loss = losses['e_latent_loss']
        print("loss", loss)

        # negative log-likelihood for a tensor of size (batch x 1 x n_output)
        #print("recons", reconstructed_x.shape, reconstructed_x.squeeze().shape, target.shape)
        #loss = F.nll_loss(reconstructed_x.squeeze(), target)

        optimizer.zero_grad()
        #loss.backward()
        optimizer.step()

        # print training stats
        if batch_idx % log_interval == 0:
            print(f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss:.6f}")

        # update progress bar
        pbar.update(pbar_update)
        # record loss
        #losses.append(loss.item())
        losses_over_epochs.append(loss)

def number_of_correct(pred, target):
    # count number of correct predictions
    return pred.squeeze().eq(target).sum().item()


def get_likely_index(tensor):
    # find most likely label index for each element in the batch
    return tensor.argmax(dim=-1)


def test(model, epoch):
    model.eval()
    correct = 0
    for data, target, speaker_id in test_loader:
        print(data.shape)
        data = data.to(device_config.device)
        target = target.to(device_config.device)

        # apply transform and model on whole batch directly on device
        data = transform(data)
        print(data.shape)
        output = model(data, speaker_dic, speaker_id)

        pred = get_likely_index(output)
        correct += number_of_correct(pred, target)

        # update progress bar
        pbar.update(pbar_update)

    print(f"\nTest Epoch: {epoch}\tAccuracy: {correct}/{len(test_loader.dataset)} ({100. * correct / len(test_loader.dataset):.0f}%)\n")

log_interval = 20
n_epoch = CONFIG['epochs']

pbar_update = 1 / (len(train_loader) + len(test_loader))
losses_over_epochs = []

# The transform needs to live on the same device as the model and the data.
transform = transform.to(device_config.device)
with tqdm(total=n_epoch) as pbar:
    for epoch in range(1, n_epoch + 1):
        train(model, epoch, log_interval)
        test(model, epoch)
        scheduler.step()


def predict(tensor):
    # Use the model to predict the label of the waveform
    tensor = tensor.to(device_config.device)
    tensor = transform(tensor)
    tensor = model(tensor.unsqueeze(0))
    tensor = get_likely_index(tensor)
    tensor = index_to_label(tensor.squeeze())
    return tensor


waveform, sample_rate, utterance, *_ = train_set[-1]

print(f"Expected: {utterance}. Predicted: {predict(waveform)}.")

for i, (waveform, sample_rate, utterance, *_) in enumerate(test_set):
    output = predict(waveform)
    if output != utterance:
        print(f"Data point #{i}. Expected: {utterance}. Predicted: {output}.")
        break
else:
    print("All examples in this dataset were correctly classified!")
    print("In this case, let's just look at the last data point")
    print(f"Data point #{i}. Expected: {utterance}. Predicted: {output}.")