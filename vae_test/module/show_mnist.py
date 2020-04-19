from __future__ import print_function
import argparse
import torch
import torch.utils.data
import torchvision
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
import custom_dataset
import numpy as np
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser(description='VAE MNIST Example')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)

device = torch.device("cuda" if args.cuda else "cpu")
"""
#train_dataset = torchvision.datasets.MNIST(
                    root='../../data',
                    train=True,
                    transform=transforms.ToTensor(),
                    download=True)

#test_dataset = torchvision.datasets.MNIST(
                    root='../../data',
                    train=False,
                    transform=transforms.ToTensor(),
                    download=True)

#kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
#train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../../data', train=True, download=True,
                   transform=transforms.ToTensor()),
    batch_size=args.batch_size, shuffle=True, **kwargs)
#test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../../data', train=False, transform=transforms.ToTensor()),
    batch_size=args.batch_size, shuffle=False, **kwargs)
"""


to_tenser_transforms = transforms.Compose([
transforms.ToTensor() # Tensorに変換
])
train_dataset = custom_dataset.CustomDataset("/home/is0383kk/workspace/study/datasets/MNIST",to_tenser_transforms,train=True)

n_samples = len(train_dataset) # n_samples is 60000
train_size = int(len(train_dataset) * 0.8) # train_size is 48000
val_size = n_samples - train_size # val_size is 48000

# shuffleしてから分割してくれる.
train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])

print(len(train_dataset)) # 48000
print(len(val_dataset)) # 12000
#print(train_dataset[0])


def show(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg,(1,2,0)),interpolation="nearest")
    plt.show()

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=10, shuffle=False)
test_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=10, shuffle=False)

for i,(images,labels) in enumerate(train_loader):
    print("i->",i)
    print("images->",images[i].size())
    print("labels->",labels[i])
    #print(labels.numpy())
    #print(type(images[0].numpy()))

    show(images[i])
    #show(torchvision.utils.make_grid(images,padding=1))
    plt.axis("off")

    #break
