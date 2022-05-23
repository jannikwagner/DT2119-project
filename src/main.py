import random
from turtle import backward
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
    rec_losses = []
    kl_losses = []
    clazz_losses = []

    for batch_idx, (audio, label, speaker_id) in enumerate(dataloader):
        audio = audio.to(config.device)  # batch_size, n_channels, n_samples
        label = label.to(config.device)
        speaker_id = speaker_id.to(config.device)
        # print(label.shape, speaker_id.shape)
        # print(label, speaker_id)
        mel_spectrogram = transform(audio)  # batch_size, n_channels, n_mel, n_windows
        optimizer.zero_grad()
        
        rec_mel_spectrogram, kl, *clazz = model(mel_spectrogram, label)  # 

        classify_loss = torch.as_tensor(0)
        if config.classify:
            clazz = clazz[0]
            classify_loss = F.binary_cross_entropy(clazz, label)
        if config.sqrt_for_loss:  # sqrt before loss
            mel_spectrogram = (mel_spectrogram).sqrt()
            rec_mel_spectrogram = (rec_mel_spectrogram).sqrt()
        rec_loss = criterion(rec_mel_spectrogram, mel_spectrogram)

        loss = rec_loss + kl + classify_loss

        loss.backward()
        optimizer.step()

        rec_losses.append(rec_loss.item())
        kl_losses.append(kl.item())
        clazz_losses.append(classify_loss.item())

        if (batch_idx) % config.log_interval == 0:  
            print(f"--> train step: {batch_idx}, rec_loss: {rec_loss.item():.5f}, kl: {kl.item():.5f}, classify_loss: {classify_loss.item():.5f}, time {time.time()-t:.5f}")
    return rec_losses, kl_losses, clazz_losses

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
    total_rec_losses = []
    total_kl_losses = []
    total_clazz_losses = []
    transform = transform.to(config.device)
    for epoch in tqdm(range(1, config.epochs + 1)):
        rec_losses, kl_losses, clazz_losses = train_one_epoch(model, train_loader, optimizer, transform, config, criterion)
        print(f"--> epoch: {epoch}, rec_loss: {np.mean(rec_losses):.5f}, kl: {np.mean(kl_losses):.5f}, clazz: {np.mean(clazz_losses):.5f}")
        if any(map(lambda x : np.any(np.isnan(x)), [rec_losses, kl_losses, clazz_losses])):
            print("LOSS IS NAN, interrupting")
            break

        total_rec_losses.extend(rec_losses)
        total_kl_losses.extend(kl_losses)
        total_clazz_losses.extend(clazz_losses)
        # test(model, epoch)
        scheduler.step()
        torch.save(model, config.TRAINED_MODEL_PATH)  # save every epoch in case of failure (TODO: should save he cpu version so that it can be loaded from cpu)

        plot(total_rec_losses, total_kl_losses, total_clazz_losses, config)  

# pass through model
def reconstruct_audio_test(model, config, train_set, transform, inverse_transform):
    print("reconstruct test")
    waveform, sample_rate, label, speaker_id, utterance_number = train_set[random.randint(0,data_manager.n_labels-1)]
    waveform = waveform.to(config.device)[None, ...]
    transform, inverse_transform = transform.to(config.device), inverse_transform.to(config.device)
    transformed = transform(waveform)
    # print("transformed")
    # print(transformed)
    # print(transformed.min(), transformed.max())
    model = model.eval()
    label_one_hot = data_manager.label_to_one_hot(label)[None, ...].to(config.device)
    with torch.no_grad():
        rec_features, *_ = model(transformed, label_one_hot)
    # print("rec_features")
    # print(rec_features)
    # print(rec_features.min(), rec_features.max())
    # print("diff")
    # print(transformed - rec_features)
    rec_wav = inverse_transform(rec_features)
    # print("rec_wav")
    # print(rec_wav.min(), rec_wav.max())
    # print(rec_wav.shape)
    torchaudio.save(os.path.join(config.EXPERIMENT_PATH, "model_rec.wav"), rec_wav.to("cpu")[0], sample_rate)
    torchaudio.save(os.path.join(config.AUDIO_PATH, "original.wav"), rec_wav.to("cpu")[0], sample_rate)

def reconstruct_normed_audio_test(config, train_set, transform, inverse_transform):
    """I just tested whether it makes a difference if mel_spectrogram files are normalized: It does, don't normalize!

    Args:
        config (_type_): _description_
        train_set (_type_): _description_
        transform (_type_): _description_
        inverse_transform (_type_): _description_
    """
    waveform, sample_rate, label, speaker_id, utterance_number = train_set[0]
    waveform = waveform.to(config.device)[None, ...]
    transform, inverse_transform = transform.to(config.device), inverse_transform.to(config.device)
    transformed = transform(waveform)
    rec_wav = inverse_transform(transformed)
    transformed_norm = transformed / transformed.max()
    rec_norm_wav = inverse_transform(transformed_norm)
    torchaudio.save(os.path.join(config.AUDIO_PATH, "rec.wav"), rec_wav.to("cpu")[0], sample_rate)
    torchaudio.save(os.path.join(config.AUDIO_PATH, "rec_norm.wav"), rec_norm_wav.to("cpu")[0], sample_rate)
    torchaudio.save(os.path.join(config.AUDIO_PATH, "original.wav"), waveform.to("cpu")[0], sample_rate)

def plot(total_rec_losses, total_kl_losses, total_clazz_losses, config):
    n = len(total_rec_losses)
    plt.plot(range(n), total_rec_losses, label="rec_loss")
    plt.plot(range(n), total_kl_losses, label="kl_loss")
    plt.plot(range(n), total_clazz_losses, label="clazz_loss")
    plt.xlabel("updates")
    plt.ylabel("loss")
    plt.legend()
    plt.savefig(os.path.join(config.EXPERIMENT_PATH, "loss.png"))
    plt.cla()

def sample_test(model, config, data_manager):
    print("sample")
    model = model.eval().to(config.device)
    label_one_hot = data_manager.get_one_hot(random.randint(0, data_manager.n_labels-1), data_manager.n_labels)[None,...].to(config.device)
    with torch.no_grad():
        sample_features = model.sample(1, label_one_hot)
    sample_wav = data_manager.inverse_transform(sample_features).to("cpu")
    print(sample_wav.shape)
    torchaudio.save(os.path.join(config.EXPERIMENT_PATH, "sample.wav"), sample_wav.to("cpu")[0], data_manager.sample_rate)

def get_acc(clazz, label):
    label = torch.argmax(label, dim=1)
    clazz = torch.argmax(clazz, dim=1)
    acc = torch.sum(clazz == label)/len(clazz)
    return acc

def classify(model, data_loader):
    model = model.eval().to(config.device)
    with torch.no_grad():
        total_loss = 0
        total_acc = 0
        for batch_idx, (audio, label, speaker_id) in enumerate(data_loader):
            audio = audio.to(config.device)
            label = label.to(config.device)
            features = data_manager.transform(audio)
            clazz = model.classify(features)
            loss = F.cross_entropy(clazz, label)
            acc = get_acc(clazz, label)
            # print("loss:", loss, "acc:", acc)
            total_loss, total_acc = total_loss + loss, total_acc + acc

    loss, acc = total_loss /(batch_idx+1), total_acc /(batch_idx+1)
    return loss, acc

if __name__ == "__main__":
    experiments = ['exp22.yaml', "exp23.yaml"]
    for experiment in experiments:
        print(experiment)
        configuration_path = 'configurations' + os.sep + experiment
        config = Config(configuration_path)

        print("config", config.config)
        print("device:", config.device)
        os.makedirs(config.DATA_DOWNLOAD_PATH, exist_ok=True)
        os.makedirs(config.AUDIO_PATH, exist_ok=True)
        os.makedirs(config.EXPERIMENT_PATH, exist_ok=True)

        # The transform needs to live on the same device as the model and the data.
        data_manager = DataManager(config)

        if config.should_train_model:
            model = get_model(config, data_manager.data_dim, data_manager.condition_dim, data_manager.n_labels)
            pytorch_total_params = sum(p.numel() for p in model.parameters())
            print(model)
            print("params:", pytorch_total_params)
            optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config.lr_step_size, gamma=config.lr_gamma)  # reduce the learning after 20 epochs by a factor of 10
            criterion = nn.MSELoss()
            train(model, optimizer, scheduler, criterion, config, data_manager.transform, data_manager.train_loader)
        else:
            model = torch.load(config.TRAINED_MODEL_PATH, map_location=config.device)
            print("trained model loaded")

        reconstruct_audio_test(model.to(config.device), config, data_manager.train_set, data_manager.transform, data_manager.inverse_transform)

        sample_test(model, config, data_manager)

        if config.classify:
            loss, acc = classify(model, data_manager.test_loader)
            with open(os.path.join(config.EXPERIMENT_PATH, "test.txt"), "w") as f:
                f.writelines([
                    f"test classify loss: {loss}\n",
                    f"test acc: {acc}"
                ])

    


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
