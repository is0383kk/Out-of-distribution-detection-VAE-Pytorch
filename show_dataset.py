import torch
import torchvision
import torch.nn as nn
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
import os
import matplotlib.pyplot as plt

dataset = torchvision.datasets.MNIST(root="./data",
                                   train=True,
                                   transform=transforms.ToTensor(),
                                   download=True)
#データローダ
data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                            batch_size=4,
                                            shuffle=True)

def show(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg,(1,2,0)),interpolation="nearest")
    plt.show()

for i in range(5):
    #バッチサイズ分だけ画像とラベルの表示
    for i,(images,labels) in enumerate(data_loader):
        print("i->",i)
        print("batch_size->"+str(images[i].size(0))+"\nheight->"+str(images[i].size(1)) + "\nwidth->" + str(images[i].size(2)))
        print("labels->",labels)
        #print(labels.numpy())
        #print(type(images[0].numpy()))

        #show(images[0])
        show(torchvision.utils.make_grid(images,padding=1))
        plt.axis("off")

        break