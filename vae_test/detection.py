from __future__ import print_function
import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from module import custom_dataset
import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='VAE MNIST Example')
parser.add_argument('--batch-size', type=int, default=1, metavar='N',
                    help='input batch size for training')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--category', type=int, default=9, metavar='K',
                    help='how many category on datesets')
parser.add_argument('--anomaly', type=int, default=0, metavar='K',
                    help='how many category on datesets')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)  

device = torch.device("cuda" if args.cuda else "cpu")

"""
kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.ToTensor()),
    batch_size=args.batch_size, shuffle=True, **kwargs)

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, download=True, transform=transforms.ToTensor()),
    batch_size=args.batch_size, shuffle=False, **kwargs)
"""
# 独自に定義したデータローダの設定
to_tenser_transforms = transforms.Compose([
transforms.ToTensor() # Tensorに変換
])
train_test_dataset = custom_dataset.CustomDataset("/home/is0383kk/workspace/study/datasets/MNIST",to_tenser_transforms,train=True)


# shuffleしてから分割してくれる.
n_samples = len(train_test_dataset) # n_samples is 60000
train_size = int(len(train_test_dataset) * 0.88) # train_size is 48000
test_size = n_samples - train_size # val_size is 48000
train_dataset, test_dataset = torch.utils.data.random_split(train_test_dataset, [train_size, test_size])



train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                            batch_size=args.batch_size,
                                            shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                            batch_size=args.batch_size,
                                            shuffle=True)
anomaly_dataset = custom_dataset.CustomDataset("/home/is0383kk/workspace/study/datasets/MNIST",to_tenser_transforms,train=False)
anomaly_loader = torch.utils.data.DataLoader(dataset=anomaly_dataset,
                                            batch_size=args.batch_size,
                                            shuffle=True)


print(f"Train data->{len(train_dataset)}")
print(f"Test data->{len(test_dataset)}")
print(f"Anomaly data->{len(anomaly_dataset)}")

ngf = 64
ndf = 64
nc = 1 # 画像のチャンネル数

def prior(K, alpha):
    """
    Prior for the model.
    :param topics: number of topics
    :return: mean and variance tensors
    """
    # ラプラス近似で正規分布に近似
    a = torch.Tensor(1, K).float().fill_(alpha) # 1 x 50 全て1.0
    mean = a.log().t() - a.log().mean(1)
    var = ((1 - 2.0 / K) * a.reciprocal()).t() + (1.0 / K ** 2) * a.reciprocal().sum(1)
    return mean.t(), var.t() # これを事前分布に定義

class VAE_DIR(nn.Module):
    def __init__(self):
        super(VAE_DIR, self).__init__()
        self.encoder = nn.Sequential(
            # input is (nc) x 28 x 28
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 14 x 14
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 7 x 7
            nn.Conv2d(ndf * 2, ndf * 4, 3, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 4 x 4
            nn.Conv2d(ndf * 4, 1024, 4, 1, 0, bias=False),
            # nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.Sigmoid()
        )

        self.decoder = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(     1024, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 3, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2,     nc, 4, 2, 1, bias=False),
            # nn.BatchNorm2d(ngf),
            # nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            # nn.ConvTranspose2d(    ngf,      nc, 4, 2, 1, bias=False),
            # nn.Tanh()
            nn.Sigmoid()
            # state size. (nc) x 64 x 64
        )
        self.fc1 = nn.Linear(1024, 512)
        self.fc21 = nn.Linear(512, args.category)
        self.fc22 = nn.Linear(512, args.category)

        self.fc3 = nn.Linear(args.category, 512)
        self.fc4 = nn.Linear(512, 1024)

        self.lrelu = nn.LeakyReLU()
        self.relu = nn.ReLU()

        # 事前分布のパラメータを定義
        self.prior_mean, self.prior_var = map(nn.Parameter, prior(args.category, 0.3))
        self.prior_logvar = nn.Parameter(self.prior_var.log())
        self.prior_mean.requires_grad = False
        self.prior_var.requires_grad = False
        self.prior_logvar.requires_grad = False


    def encode(self, x):
        conv = self.encoder(x);
        # print("encode conv", conv.size())
        h1 = self.fc1(conv.view(-1, 1024))
        # print("encode h1", h1.size())
        return self.fc21(h1), self.fc22(h1)

    def decode(self, z):
        z = F.softmax(z,dim=1)
        h3 = self.relu(self.fc3(z))
        deconv_input = self.fc4(h3)
        # print("deconv_input", deconv_input.size())
        deconv_input = deconv_input.view(-1,1024,1,1)
        # print("deconv_input", deconv_input.size())
        return self.decoder(deconv_input)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std


    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    # Reconstruction + KL divergence losses summed over all elements and batch
    def loss_function_dir(self, recon_x, x, mu, logvar, K, beta):
        BCE = F.binary_cross_entropy(recon_x.view(-1, 784), x.view(-1, 784), reduction='sum')
        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        # KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        # 事前分布の定義
        # 事前分布のパラメータを定義
        prior_mean = self.prior_mean.expand_as(mu)
        prior_var = self.prior_var.expand_as(logvar)
        prior_logvar = self.prior_logvar.expand_as(logvar)
        var_division = logvar.exp() / prior_var # Σ_0 / Σ_1
        diff = mu - prior_mean # μ_１ - μ_0
        diff_term = diff *diff / prior_var # (μ_1 - μ_0)(μ_1 - μ_0)/Σ_1
        logvar_division = prior_logvar - logvar # log|Σ_1| - log|Σ_0| = log(|Σ_1|/|Σ_2|)
        # KL
        KLD = 0.5 * ((var_division + diff_term + logvar_division).sum(1) - K)
        #print(KLD)
        
        return (beta * BCE) + KLD, -BCE

class VAE_CNN(nn.Module):
    def __init__(self):
        super(VAE_CNN, self).__init__()
        self.encoder = nn.Sequential(
            # input is (nc) x 28 x 28
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 14 x 14
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 7 x 7
            nn.Conv2d(ndf * 2, ndf * 4, 3, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 4 x 4
            nn.Conv2d(ndf * 4, 1024, 4, 1, 0, bias=False),
            # nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.Sigmoid()
        )

        self.decoder = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(     1024, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 3, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2,     nc, 4, 2, 1, bias=False),
            # nn.BatchNorm2d(ngf),
            # nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            # nn.ConvTranspose2d(    ngf,      nc, 4, 2, 1, bias=False),
            # nn.Tanh()
            nn.Sigmoid()
            # state size. (nc) x 64 x 64
        )
        self.fc1 = nn.Linear(1024, 512)
        self.fc21 = nn.Linear(512, 64)
        self.fc22 = nn.Linear(512, 64)

        self.fc3 = nn.Linear(64, 512)
        self.fc4 = nn.Linear(512, 1024)

        self.lrelu = nn.LeakyReLU()
        self.relu = nn.ReLU()

    def encode(self, x):
        conv = self.encoder(x);
        # print("encode conv", conv.size())
        h1 = self.fc1(conv.view(-1, 1024))
        # print("encode h1", h1.size())
        return self.fc21(h1), self.fc22(h1)

    def decode(self, z):
        h3 = self.relu(self.fc3(z))
        deconv_input = self.fc4(h3)
        # print("deconv_input", deconv_input.size())
        deconv_input = deconv_input.view(-1,1024,1,1)
        # print("deconv_input", deconv_input.size())
        return self.decoder(deconv_input)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std


    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar
    
    # Reconstruction + KL divergence losses summed over all elements and batch
    def loss_function_cnn(self, recon_x, x, mu, logvar, beta):
        BCE = F.binary_cross_entropy(recon_x.view(-1, 784), x.view(-1, 784), reduction='sum')

        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        return (beta * BCE) + KLD, -BCE


model_dir = VAE_DIR().to(device)
model_dir.load_state_dict(torch.load('./pth/dir_vae'+str(args.anomaly)+'.pth'))
model_dir.eval()

model_cnn = VAE_CNN().to(device)
model_cnn.load_state_dict(torch.load('./pth/cnn_vae'+str(args.anomaly)+'.pth'))
model_cnn.eval()

model_dir_beta = VAE_DIR().to(device)
model_dir_beta.load_state_dict(torch.load('./pth/dir_vae'+str(args.anomaly)+'_b.pth'))
model_dir_beta.eval()

model_cnn_beta = VAE_CNN().to(device)
model_cnn_beta.load_state_dict(torch.load('./pth/cnn_vae'+str(args.anomaly)+'_b.pth'))
model_cnn_beta.eval()

print(model_cnn)

y_score_cnn = []
y_score_dir = []
y_score_cnn_beta = []
y_score_dir_beta = []
# CNN
for i, (data, _) in enumerate(test_loader):
    with torch.no_grad():
        data = data.to(device)
        #print(f"{i} : data => {len(data)}")
        recon_batch, mu, logvar = model_cnn(data)
        recon_batch_b, mu_b, logvar_b = model_cnn_beta(data)
        loss, BCE = model_cnn.loss_function_cnn(recon_batch, data, mu, logvar, 1.0)
        loss_b, BCE_b = model_cnn_beta.loss_function_cnn(recon_batch_b, data, mu_b, logvar_b, 10.0)
        loss = loss.cpu().detach().numpy()
        loss = np.round(loss, 1)
        loss_b = loss_b.cpu().detach().numpy()
        loss_b = np.round(loss_b, 1)
        y_score_cnn.append(loss)
        y_score_cnn_beta.append(loss_b)
        #print(f"y_score => {y_score}")
        #print(f"cnn_loss => {cnn_loss}")

for i, (data, _) in enumerate(anomaly_loader):
    with torch.no_grad():
        data = data.to(device)
        #print(f"{i} : data => {len(data)}")
        recon_batch, mu, logvar = model_cnn(data)
        recon_batch_b, mu_b, logvar_b = model_cnn_beta(data)
        loss, BCE = model_cnn.loss_function_cnn(recon_batch, data, mu, logvar, 1.0)
        loss_b, BCE_b = model_cnn_beta.loss_function_cnn(recon_batch_b, data, mu_b, logvar_b, 10.0)
        loss = loss.cpu().detach().numpy()
        loss_b = loss_b.cpu().detach().numpy()
        loss = np.round(loss, 1)
        loss_b = np.round(loss_b, 1)
        y_score_cnn.append(loss)
        y_score_cnn_beta.append(loss_b)
        #print(f"y_score => {y_score}")

# Dir
for i, (data, _) in enumerate(test_loader):
    with torch.no_grad():
        data = data.to(device)
        #print(f"{i} : data => {len(data)}")
        recon_batch, mu, logvar = model_dir(data)
        recon_batch_b, mu_b, logvar_b = model_dir_beta(data)
        loss, BCE = model_dir.loss_function_dir(recon_batch, data, mu, logvar, args.category, 1.0)
        loss_b, BCE_b = model_dir_beta.loss_function_dir(recon_batch_b, data, mu_b, logvar_b, args.category, 10.0)
        loss = loss.cpu().detach().numpy()
        loss_b = loss_b.cpu().detach().numpy()
        loss = np.round(loss, 1)
        loss_b = np.round(loss_b, 1)
        y_score_dir.append(loss)
        y_score_dir_beta.append(loss_b)
        #print(f"y_score => {y_score}")
        #print(f"cnn_loss => {cnn_loss}")

for i, (data, _) in enumerate(anomaly_loader):
    with torch.no_grad():
        data = data.to(device)
        #print(f"{i} : data => {len(data)}")
        recon_batch, mu, logvar = model_dir(data)
        recon_batch_b, mu_b, logvar_b = model_dir_beta(data)
        loss, BCE = model_dir.loss_function_dir(recon_batch, data, mu, logvar, args.category,1.0)
        loss_b, BCE_b = model_dir_beta.loss_function_dir(recon_batch_b, data, mu_b, logvar_b, args.category,10.0)
        loss = loss.cpu().detach().numpy()
        loss = np.round(loss, 1)
        loss_b = loss_b.cpu().detach().numpy()
        loss_b = np.round(loss_b, 1)
        y_score_dir.append(loss)
        y_score_dir_beta.append(loss_b)
        #print(f"y_score => {y_score}")


import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt

#y_true = np.array([0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1])
test_label = np.zeros(len(test_dataset))
anomaly_lable = np.ones(len(anomaly_dataset))
y_true = np.concatenate([test_label, anomaly_lable])

print(f"y_score_cnn => {len(y_score_cnn)}")
print(f"y_score_dir => {len(y_score_dir)}")
print(f"y_true => {len(y_true)}")

fpr_cnn, tpr_cnn, thresholds_cnn = metrics.roc_curve(y_true, y_score_cnn)
fpr_dir, tpr_dir, thresholds_dir = metrics.roc_curve(y_true, y_score_dir)
fpr_cnn_beta, tpr_cnn_beta, thresholds_cnn_beta = metrics.roc_curve(y_true, y_score_cnn_beta)
fpr_dir_beta, tpr_dir_beta, thresholds_dir_beta = metrics.roc_curve(y_true, y_score_dir_beta)
#print(f"thresholds => {thresholds}")
auc_cnn = metrics.auc(fpr_cnn, tpr_cnn)
auc_dir = metrics.auc(fpr_dir, tpr_dir)
auc_cnn_beta = metrics.auc(fpr_cnn_beta, tpr_cnn_beta)
auc_dir_beta = metrics.auc(fpr_dir_beta, tpr_dir_beta)
print(f"AUC_CNN => {auc_cnn}")
print(f"AUC_Dir => {auc_dir}")
print(f"AUC_CNN_B => {auc_cnn_beta}")
print(f"AUC_Dir_B => {auc_dir_beta}")
l1, l2, l3, l4 = "Baseline", "Proposed", "Baseline(b=10)", "Proposed(b=10)"
plt.plot(fpr_cnn, tpr_cnn, label=l1)
plt.plot(fpr_dir, tpr_dir, label=l2)
plt.plot(fpr_cnn_beta, tpr_cnn_beta, label=l3)
plt.plot(fpr_dir_beta, tpr_dir_beta, label=l4)
plt.legend(loc='lower right')
plt.xlabel('FPR: False positive rate',fontsize=13)
plt.ylabel('TPR: True positive rate',fontsize=13)
plt.grid()
plt.savefig('roc'+str(args.anomaly)+'.png')
plt.show()
