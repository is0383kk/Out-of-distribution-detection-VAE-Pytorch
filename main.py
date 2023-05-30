from __future__ import print_function
import os
import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics

os.makedirs("./recon", exist_ok=True)

# Anomaly number
ANOMALY_TARGET = 5

parser = argparse.ArgumentParser(description='VAE MNIST Example')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=20, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--no-mps', action='store_true', default=False,
                        help='disables macOS GPU training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
use_mps = not args.no_mps and torch.backends.mps.is_available()

torch.manual_seed(args.seed)

if args.cuda:
    device = torch.device("cuda")
elif use_mps:
    device = torch.device("mps")
else:
    device = torch.device("cpu")

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

# Training dataset excluding anomaly target numbers
train_dataset = datasets.MNIST('../data', train=True, download=True, transform=transforms.ToTensor())
train_mask = (train_dataset.targets != ANOMALY_TARGET)
train_dataset.data = train_dataset.data[train_mask]
train_dataset.targets = train_dataset.targets[train_mask]
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)
all_train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True, **kwargs)


# anomaly target numeric-only dataset
anomaly_dataset = datasets.MNIST('../data', train=False, download=True, transform=transforms.ToTensor())
anomaly_mask = (anomaly_dataset.targets == ANOMALY_TARGET)
anomaly_dataset.data = anomaly_dataset.data[anomaly_mask]
anomaly_dataset.targets = anomaly_dataset.targets[anomaly_mask]
anomaly_loader = torch.utils.data.DataLoader(anomaly_dataset, batch_size=args.batch_size, shuffle=False, **kwargs)
all_anomaly_loader = torch.utils.data.DataLoader(anomaly_dataset, batch_size=1, shuffle=False, **kwargs)


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        self.fc1 = nn.Linear(784, 400)
        self.fc21 = nn.Linear(400, 20)
        self.fc22 = nn.Linear(400, 20)
        self.fc3 = nn.Linear(20, 400)
        self.fc4 = nn.Linear(400, 784)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


model = VAE().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)


# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD


def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item() / len(data)))
            
        # Reconstruction for training data
        if batch_idx == 0:
                n = min(data.size(0), 8)
                comparison = torch.cat([data[:n], recon_batch.view(args.batch_size, 1, 28, 28)[:n]])
                save_image(comparison.cpu(), 'recon/recon_train' + str(epoch) + '.png', nrow=n)

    print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss / len(train_loader.dataset)))

    return - (train_loss / len(train_loader.dataset))


def anomaly(epoch):
    model.eval()
    anomaly_loss = 0
    with torch.no_grad():
        for i, (data, _) in enumerate(anomaly_loader):
            data = data.to(device)
            recon_batch, mu, logvar = model(data)
            anomaly_loss += loss_function(recon_batch, data, mu, logvar).item()

            # Reconstruction for anomaly data
            if i == 0:
                n = min(data.size(0), 8)
                comparison = torch.cat([data[:n], recon_batch.view(args.batch_size, 1, 28, 28)[:n]])
                save_image(comparison.cpu(), 'recon/recon_anomaly' + str(epoch) + '.png', nrow=n)

    anomaly_loss /= len(anomaly_loader.dataset)
    print('====> anomaly set loss: {:.4f}'.format(anomaly_loss))

    return - anomaly_loss

def plot_roc():
    y_true = np.concatenate([np.zeros(len(train_dataset)), np.ones(len(anomaly_dataset))])
    y_score = []
    model.eval()
    with torch.no_grad():
        for i, (data, _) in enumerate(all_train_loader):
            data = data.to(device)
            recon_batch, mu, logvar = model(data)
            train_score_loss = loss_function(recon_batch, data, mu, logvar)
            train_score_loss = train_score_loss.cpu()
            y_score.append(np.round(train_score_loss, 1).detach().numpy())
        for i, (data, _) in enumerate(all_anomaly_loader):
            data = data.to(device)
            recon_batch, mu, logvar = model(data)
            anomaly_score_loss = loss_function(recon_batch, data, mu, logvar)
            anomaly_score_loss = anomaly_score_loss.cpu()
            y_score.append(np.round(anomaly_score_loss, 1).detach().numpy())

    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_score)
    auc = metrics.auc(fpr, tpr)
    plt.plot(fpr, tpr)
    plt.xlabel('FPR: False positive rate', fontsize=13); plt.ylabel('TPR: True positive rate', fontsize=13)
    plt.grid()
    plt.savefig('./roc'+str(ANOMALY_TARGET)+'.png')
    plt.close()
    print("AUC:" + str(np.round(auc, 2)))

if __name__ == "__main__":
    fig1, ax1 = plt.subplots()
    ax1.set_xlabel('Epoch', fontsize=15); ax1.set_ylabel('ELBO', fontsize=15)  
    train_elbo_list = [];anomaly_loss_list = []

    for epoch in range(1, args.epochs + 1):
        avg_train_elbo = train(epoch);avg_anomaly_elbo = anomaly(epoch)
        train_elbo_list.append(avg_train_elbo);anomaly_loss_list.append(avg_anomaly_elbo)

    # Plot ELBO
    ax1.plot(np.arange(args.epochs), train_elbo_list, color="blue", label="ELBO_Train")
    ax1.plot(np.arange(args.epochs), anomaly_loss_list, color="red", label="ELBO_Anomaly")
    fig1.savefig('./elbo.png')
    plt.close()

    # Plot ROC
    plot_roc()