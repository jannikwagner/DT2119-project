from distutils.command.config import config
from tqdm import tqdm
import torchaudio
from torch import nn
import torch.nn.functional as F
import time
import torch

from data import data_dim, train_loader, test_loader, transform, transform_InverseMelScale, transform_GriffinLim, speaker_dic, transform_MelSpectrogram, train_set
from config import config


##############################################
# CREATE MODEL
###############################################
def mc_kl_divergence(self, z, mu, std):
    # stolen from https://towardsdatascience.com/variational-autoencoder-demystified-with-pytorch-implementation-3a06bee395ed
    # --------------------------
    # Monte carlo KL divergence
    # --------------------------
    # 1. define the first two probabilities (in this case Normal for both)
    p = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std))
    q = torch.distributions.Normal(mu, std)

    # 2. get the probabilities from the equation
    log_qzx = q.log_prob(z)
    log_pz = p.log_prob(z)

    # kl
    kl = (log_qzx - log_pz)
    
    # sum over last dim to go from single dim distribution to multi-dim
    kl = kl.sum(-1)
    return kl

class VariationalEncoder(nn.Module):
    def __init__(self, data_dim, latent_dim):
        super(VariationalEncoder, self).__init__()
        self.latent_dim = latent_dim
        self.data_dim = data_dim
        print(data_dim)
        self.data_dim_flat = torch.prod(torch.as_tensor(data_dim))
        print(self.data_dim_flat)

        layers = [
            torch.nn.Flatten(),
            nn.Linear(self.data_dim_flat, 512),
            nn.BatchNorm1d(512),
            nn.ReLU()
        ]
        self.seq = nn.Sequential(*layers)

        self.fc_z_mu = nn.Linear(512, latent_dim)
        self.fc_z_log_var = nn.Linear(512, latent_dim)
        
        self.N = torch.distributions.Normal(0, 1)
        self.N.loc = self.N.loc.to(config.device)
        self.N.scale = self.N.scale.to(config.device)
    
    def forward(self, x):
        #x = torch.flatten(x, start_dim=1)
        # x = torch.flatten(x, 1)
        x = self.seq(x)
        z_mu = self.fc_z_mu(x)
        z_log_var = self.fc_z_log_var(x)
        z_var = torch.exp(z_log_var)
        z_std = torch.exp(z_log_var/2)
        z = z_mu + z_std*self.N.sample(z_mu.shape)
        kl = torch.mean(-0.5 * torch.sum(1 + z_log_var - z_mu ** 2 - z_var, dim = 1), dim = 0)
        return z, kl


class Decoder(nn.Module):
    def __init__(self, data_dim, latent_dim):
        super(Decoder, self).__init__()
        self.latent_dim = latent_dim
        self.data_dim = data_dim
        self.data_dim_flat = torch.prod(torch.as_tensor(data_dim))
        layers = [
            nn.Linear(latent_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, self.data_dim_flat),
            torch.nn.Unflatten(1, self.data_dim)
        ]
        self.seq = nn.Sequential(*layers)
        
    def forward(self, z):
        z = self.seq(z)
        return z


class VariationalAutoencoder(nn.Module):
    def __init__(self, data_dim, latent_dim):
        super(VariationalAutoencoder, self).__init__()
        self.latent_dim = latent_dim
        self.data_dim = data_dim
        self.data_dim_flat = torch.prod(torch.as_tensor(data_dim))
        self.encoder = VariationalEncoder(data_dim, latent_dim)
        self.decoder = Decoder(data_dim, latent_dim)

    def forward(self, x):
        z, kl = self.encoder(x)
        return self.decoder(z), kl

model = VariationalAutoencoder(data_dim, config.latent_dim).to(config.device)


##############################################
# TRAINING
##############################################

import time
def train_one_epoch(model, dataloader, optimizer, transform, device, criterion):
    model.train()
    t = time.time()
    total_loss = 0

    for batch_idx, (audio, label, speaker_id) in enumerate(dataloader):
        audio = audio.to(device)  # batch_size, n_channels, n_samples
        label = label.to(device)
        speaker_id = speaker_id.to(device)
        mel_spectrogram = transform(audio)  # batch_size, n_channels, n_mel, n_windows
        
        rec_mel_spectrogram, kl = model(mel_spectrogram)  # 

        optimizer.zero_grad()
        rec_loss = criterion(rec_mel_spectrogram, mel_spectrogram)
        loss = rec_loss + kl
        loss.backward()
        optimizer.step()

        batchsize = audio.size()[0]
        total_loss += loss.detach().cpu().numpy().sum()/batchsize

        if (batch_idx+1) % 10 == 0:    
            print(f"--> train step: {batch_idx}, rec_loss: {rec_loss.detach().cpu().numpy().sum()/batchsize:.5f}, kl: {kl.detach().cpu().numpy().sum()/batchsize:.5f}, time {time.time()-t:.5f}")
    
    return total_loss/len(dataloader)


optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0.0001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)  # reduce the learning after 20 epochs by a factor of 10
criterion = nn.MSELoss()


def number_of_correct(pred, target):
    # count number of correct predictions
    return pred.squeeze().eq(target).sum().item()


def get_likely_index(tensor):
    # find most likely label index for each element in the batch
    print("tensor", tensor)
    return tensor.argmax(dim=-1)


def test(model, epoch):
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

log_interval = 20
n_epoch = config.epochs
loss_over_epochs = []


# The transform needs to live on the same device as the model and the data.
if config.should_train_model:
    transform = transform.to(config.device)
    for epoch in tqdm(range(1, n_epoch + 1)):
        loss = train_one_epoch(model, train_loader, optimizer, transform, config.device, criterion)
        loss_over_epochs.append(loss)
        # test(model, epoch)
        scheduler.step()
    torch.save(model, config.TRAINED_MODEL_PATH)
else:
    model = torch.load(config.TRAINED_MODEL_PATH)
print("trained model loaded")

# pass through model

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
torchaudio.save(config.AUDIO_PATH + "model_rec.wav", rec_wav[0], sample_rate)


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