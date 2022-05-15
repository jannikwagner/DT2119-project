from tqdm import tqdm
import torchaudio
from torch import nn
import torch.nn.functional as F
import time
import torch
import os
import matplotlib.pyplot as plt
import numpy as np

from data import DataManager
from configuration import Config
from model import get_model


##############################################
# CREATE MODEL
###############################################


##############################################
# TRAINING
##############################################

import time
def train_one_epoch(model, dataloader, optimizer, transform, config, criterion):
    model = model.train()
    t = time.time()
    total_rec_loss = 0
    total_kl_loss = 0

    for batch_idx, (audio, label, speaker_id) in enumerate(dataloader):
        audio = audio.to(config.device)  # batch_size, n_channels, n_samples
        label = label.to(config.device)
        speaker_id = speaker_id.to(config.device)
        mel_spectrogram = transform(audio)  # batch_size, n_channels, n_mel, n_windows
        
        rec_mel_spectrogram, kl = model(mel_spectrogram)  # 

        optimizer.zero_grad()
        rec_loss = criterion(rec_mel_spectrogram, mel_spectrogram)
        loss = rec_loss + kl
        loss.backward()
        optimizer.step()

        batchsize = audio.size()[0]
        total_rec_loss += rec_loss.detach().cpu().numpy().sum()
        total_kl_loss += kl.detach().cpu().numpy().sum()

        if (batch_idx+1) % config.log_interval == 0:  
            print(f"--> train step: {batch_idx}, rec_loss: {rec_loss.detach().cpu().numpy().sum()/batchsize:.5f}, kl: {kl.detach().cpu().numpy().sum()/batchsize:.5f}, time {time.time()-t:.5f}")
    return total_rec_loss/len(dataloader), total_kl_loss/len(dataloader)

def number_of_correct(pred, target):
    # count number of correct predictions
    return pred.squeeze().eq(target).sum().item()

def get_likely_index(tensor):
    # find most likely label index for each element in the batch
    print("tensor", tensor)
    return tensor.argmax(dim=-1)

def test_one_epoch(model, epoch, test_loader, transform, speaker_dic):  # currently not used
    model.eval()
    correct = 0
    for data, target, speaker_id in test_loader:
        data = data.to(config.device)
        target = target.to(config.device)

        # apply transform and model on whole batch directly on device
        data = transform(data)
        output = model(data, speaker_dic, speaker_id)

        pred = get_likely_index(output)
        correct += number_of_correct(pred, target)

        # update progress bar

    print(f"\nTest Epoch: {epoch}\tAccuracy: {correct}/{len(test_loader.dataset)} ({100. * correct / len(test_loader.dataset):.0f}%)\n")

def train(model, optimizer, scheduler, criterion, config, transform, train_loader):
    rec_loss_over_epochs = []
    kl_loss_over_epochs = []
    transform = transform.to(config.device)
    for epoch in tqdm(range(1, config.epochs + 1)):
        rec_loss, kl_loss = train_one_epoch(model, train_loader, optimizer, transform, config, criterion)
        print(f"--> epoch: {epoch}, rec_loss: {rec_loss:.5f}, kl: {kl_loss:.5f}")
        if np.isnan(rec_loss) or np.isnan(kl_loss):
            print("LOSS IS NAN, interrupting")
            break

        rec_loss_over_epochs.append(rec_loss)
        kl_loss_over_epochs.append(kl_loss)
        # test(model, epoch)
        scheduler.step()
        torch.save(model, config.TRAINED_MODEL_PATH)  # save every epoch in case of failure (TODO: should save he cpu version so that it can be loaded from cpu)
    return rec_loss_over_epochs, kl_loss_over_epochs

# pass through model
def reconstruct_audio_test(model, config, train_set, transform_MelSpectrogram, transform_InverseMelScale, transform_GriffinLim):
    waveform, sample_rate, label, speaker_id, utterance_number = train_set[0]
    waveform = waveform.to(config.device)
    transform_MelSpectrogram, transform_InverseMelScale, transform_GriffinLim = transform_MelSpectrogram.to(config.device), transform_InverseMelScale.to(config.device), transform_GriffinLim.to(config.device)
    transformed = transform_MelSpectrogram(waveform)
    print("transformed")
    print(transformed)
    print(transformed.min(), transformed.max())
    model = model.eval()
    with torch.no_grad():
        rec_melspec, _ = model(transformed)
    print("rec_melspec")
    print(rec_melspec)
    print(rec_melspec.min(), rec_melspec.max())
    print("diff")
    print(transformed - rec_melspec)
    rec_spec = transform_InverseMelScale(rec_melspec)
    print("rec_spec")
    print(rec_spec.shape)
    print(rec_spec.min(), rec_spec.max())
    rec_wav = transform_GriffinLim(rec_spec)
    print("rec_wav")
    print(rec_wav.min(), rec_wav.max())
    print(rec_wav.shape)
    torchaudio.save(os.path.join(config.EXPERIMENT_PATH, "model_rec.wav"), rec_wav.to("cpu")[0], sample_rate)
    torchaudio.save(os.path.join(config.AUDIO_PATH, "original.wav"), rec_wav.to("cpu")[0], sample_rate)

def plot(rec_loss_over_epochs, kl_loss_over_epochs, config):
    n = len(rec_loss_over_epochs)
    plt.plot(range(n), rec_loss_over_epochs, label="rec_loss")
    plt.plot(range(n), kl_loss_over_epochs, label="kl_loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.savefig(os.path.join(config.EXPERIMENT_PATH, "loss.png"))

def config_init(config):
    config

if __name__ == "__main__":
    configuration_path = 'configurations' + os.sep + 'exp5.yaml'
    config = Config(configuration_path)

    print("config", config.config)
    print("device:", config.device)
    os.makedirs(config.DATA_DOWNLOAD_PATH, exist_ok=True)
    os.makedirs(config.AUDIO_PATH, exist_ok=True)
    os.makedirs(config.EXPERIMENT_PATH, exist_ok=True)

    # The transform needs to live on the same device as the model and the data.
    data_manager = DataManager(config)
    if config.should_train_model:
        model = get_model(config, data_manager.data_dim)
        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config.lr_step_size, gamma=config.lr_gamma)  # reduce the learning after 20 epochs by a factor of 10
        criterion = nn.MSELoss()
        rec_loss_over_epochs, kl_loss_over_epochs = train(model, optimizer, scheduler, criterion, config, data_manager.transform, data_manager.train_loader)
        plot(rec_loss_over_epochs, kl_loss_over_epochs, config)   
    else:
        model = torch.load(config.TRAINED_MODEL_PATH, map_location=config.device)
    print("trained model loaded")

    reconstruct_audio_test(model.to(config.device), config, data_manager.train_set, data_manager.transform_MelSpectrogram, data_manager.transform_InverseMelScale, data_manager.transform_GriffinLim)
    


# def predict(tensor):
#     # Use the model to predict the label of the waveform
#     tensor = tensor.to(DEVICE_CONFIG.device)
#     tensor = transform(tensor)
#     output = model(tensor.unsqueeze(0))
#     tensor, *_  = output
#     tensor = get_likely_index(tensor)
#     tensor = index_to_label(tensor.squeeze())
#     return tensor


# waveform, sample_rate, utterance, *_ = train_set[-1]
# print("type1", waveform.type())

# print(f"Expected: {utterance}. Predicted: {predict(waveform)}.")

# for i, (waveform, sample_rate, utterance, *_) in enumerate(test_set):
#     output = predict(waveform)
#     if output != utterance:
#         print(f"Data point #{i}. Expected: {utterance}. Predicted: {output}.")
#         break
# else:
#     print("All examples in this dataset were correctly classified!")
#     print("In this case, let's just look at the last data point")
#     print(f"Data point #{i}. Expected: {utterance}. Predicted: {output}.")
