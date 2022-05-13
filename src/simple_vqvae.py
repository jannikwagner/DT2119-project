from distutils.command.config import config
from tqdm import tqdm
import torchaudio
from torch import nn
import torch.nn.functional as F
import time
import torch
import os

from data import data_dim, train_loader, test_loader, transform, transform_InverseMelScale, transform_GriffinLim, speaker_dic, transform_MelSpectrogram, train_set
from config import config
from model import get_model


##############################################
# CREATE MODEL
###############################################


##############################################
# TRAINING
##############################################

import time
def train_one_epoch(model, dataloader, optimizer, transform, config, criterion):
    model.train()
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
        total_rec_loss += loss.detach().cpu().numpy().sum()
        total_kl_loss += loss.detach().cpu().numpy().sum()

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


def test_one_epoch(model, epoch):  # currently not used
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

def train(model, optimizer, scheduler, criterion, config, transform):
    rec_loss_over_epochs = []
    kl_loss_over_epochs = []
    transform = transform.to(config.device)
    for epoch in tqdm(range(1, config.epochs + 1)):
        rec_loss, kl_loss = train_one_epoch(model, train_loader, optimizer, transform, config, criterion)
        rec_loss_over_epochs.append(rec_loss)
        kl_loss_over_epochs.append(kl_loss)
        # test(model, epoch)
        scheduler.step()
        torch.save(model.to("cpu"), config.TRAINED_MODEL_PATH)  # save every epoch in case of failure
    return rec_loss_over_epochs, kl_loss_over_epochs



# pass through model
def reconstruct_audio_test(model, config, train_set):
    waveform, sample_rate, label, speaker_id, utterance_number = train_set[0]
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
    torchaudio.save(os.path.join(config.EXPERIMENT_PATH, "model_rec.wav"), rec_wav[0], sample_rate)
    torchaudio.save(os.path.join(config.AUDIO_PATH, "original.wav"), rec_wav[0], sample_rate)

if __name__ == "__main__":

    # The transform needs to live on the same device as the model and the data.
    if config.should_train_model:
        model = get_model(config, data_dim)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0.0001)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)  # reduce the learning after 20 epochs by a factor of 10
        criterion = nn.MSELoss()
        rec_loss_over_epochs, kl_loss_over_epochs = train(model, optimizer, scheduler, criterion, config, transform)
    else:
        model = torch.load(config.TRAINED_MODEL_PATH)
    print("trained model loaded")

    reconstruct_audio_test(model, config, train_set)



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