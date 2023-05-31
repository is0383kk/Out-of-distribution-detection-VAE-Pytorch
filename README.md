# Out-of-distribution-detection-VAE-Pytorch
This repository is for out-of-distribution detection using VAE  

# VAE features in out-of-distribution detection
VAE maximizes the objective function (ELBO) during training.  

At this time, the ELBO is calculated lower for data not used for training than for data used for training data.   
As shown in the image below.（Blue line is the data used for training・Red line is data not used for training）
<div>
	<img src='./img/example_elbo.png' height="300px">
</div>
Also, for the reconstructed images, the data used for training are reconstructed well, while the data not used for training are not reconstructed.  
This is an image reconstructed on the training data.
<div><img src='./recon/recon_train10.png' height="150px"></div>
This is an image reconstructed on data not used for training.
<div><img src='./recon/recon_anomaly10.png' height="150px"></div>
  
  
This repository uses this feature to detect out-of-distribution.  

# How to run
In the local environment, the following command can be used to execute it.
```
$ python3 main.py
```
It is easier to run "main.ipynb" with Google Colaboratory → [main.ipynb](https://colab.research.google.com/github/is0383kk/AnomalyDetection_VAE/blob/master/main.ipynb)


## Sample Output
```
$ python main.py
Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to ../data/MNIST/raw/train-images-idx3-ubyte.gz
100%|██████████| 9912422/9912422 [00:00<00:00, 100949408.99it/s]
Extracting ../data/MNIST/raw/train-images-idx3-ubyte.gz to ../data/MNIST/raw

Downloading http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz
Downloading http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz to ../data/MNIST/raw/t10k-labels-idx1-ubyte.gz
100%|██████████| 4542/4542 [00:00<00:00, 23232352.16it/s]
Extracting ../data/MNIST/raw/t10k-labels-idx1-ubyte.gz to ../data/MNIST/raw

Train Epoch: 1 [0/54579 (0%)]	Loss: 550.656433
Train Epoch: 1 [1280/54579 (2%)]	Loss: 292.733917
Train Epoch: 1 [2560/54579 (5%)]	Loss: 229.428635
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Train Epoch: 20 [52480/54579 (96%)]	Loss: 96.700661
Train Epoch: 20 [53760/54579 (98%)]	Loss: 104.435272
====> Epoch: 20 Average loss: 102.5674
====> anomaly set loss: 133.8387
AUC:0.83


```
# Evaluation of out-of-distribution detection
The results of out-of-distribution detection are evaluated by ROC curve and AUC.  
  
The result of a successful out-of-distribution detection is as follows.（The AUC in this case is 0.8）

<div><img src='./roc5.png' height="300px"></div>

- In case of good results
    - ROC curve leans to the upper left
    - AUC is higher
  
On the other hand, the results of poor out-of-distribution detection are as follows.（The AUC in this case is 0.04））

<div><img src='./roc1.png' height="300px"></div>

- In case of bad result
    - ROC curve is shifted to the lower right
    - AUC is lower
  




