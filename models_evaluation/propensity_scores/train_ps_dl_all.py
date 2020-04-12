import torch
import torch.nn as nn
from torch import tensor
from torch.nn import functional as F
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import time
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
import seaborn as sns


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.input_layer=nn.Linear(79,40)
        self.h1=nn.Linear(40,20)
        self.h2=nn.Linear(20,10)
        self.out_layer=nn.Linear(10,1)

    def learn(self, x):
        h=self.input_layer(x)
        h=self.h1(F.relu(h))
        h=self.h2(F.relu(h))
        h=self.out_layer(F.relu(h))

        return F.sigmoid(h)


class MyDataset(Dataset):
    def __init__(self, file_name, samples_type, pseudo=None):
        self.samples=pd.read_csv("../datasets/{}{}.csv".format(file_name, samples_type), index_col=0).drop(['Y', 'T', 'scores'], axis=1)
        self.labels=pd.read_csv("../datasets/{}{}.csv".format(file_name, samples_type), index_col=0).loc[:,['T']]
        self.samples=tensor(self.samples.values.astype(np.float)).float()
        self.labels=tensor(self.labels.values.astype(np.float)).float()
    def __len__(self):
        return self.samples.shape[0]

    def __getitem__(self, idx):
        return self.samples[idx], self.labels[idx]


def main(file_name):
    MOD_FACTOR=10 ** 1
    print "is CUDA available? ", torch.cuda.is_available()
    torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    dataset_train=MyDataset(file_name, "")

    loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=10,  shuffle=True, pin_memory=False)
    net=Net()
    lr = 0.001
    optimizer = optim.Adam(net.parameters(), lr=lr)
    start=time.time()
    l_running_loss_train=[]
    l_running_loss_test=[]
    n_epochs=200
    for cur_epoch in np.arange(n_epochs):

        lr = 0.001 if cur_epoch < n_epochs/2 else 0.0001
        optimizer = optim.Adam(net.parameters(), lr=lr)

        running_loss_train=0
        running_loss_test = 0
        for i, data in enumerate(loader_train, 0):
            x, y = data
            y_hat=net.learn(x)
            optimizer.zero_grad()
            loss = torch.nn.BCELoss()(y_hat, y)
            loss.backward()
            optimizer.step()
            running_loss_train+=loss.item()
        running_loss_train /= (i + 1)

        l_running_loss_train.append(running_loss_train)

        if cur_epoch %MOD_FACTOR == 0:
            print "epoch#{}: loss train: {}".format(cur_epoch, running_loss_train)
            print ""

    y_hats=tensor([])
    with torch.no_grad():
        for i, data in enumerate(loader_train, 0):
            x, y = data
            y_hat=net.learn(x)
            y_hats=torch.cat((y_hats, y_hat),0)
            optimizer.zero_grad()
            loss = torch.nn.BCELoss()(y_hat, y)
            running_loss_train+=loss.item()

    plt.clf()
    sns.distplot(y_hats.cpu().numpy(), norm_hist=False, kde=False)
    plt.savefig("../output/dl_dist_{}.png".format(file_name))

    end=time.time()
    torch.save(net.state_dict(), "../output/dl_ps_all_model_{}".format(file_name))
    print "total time: {}".format(end-start)

    plt.clf()
    plt.plot(np.arange(1, n_epochs+1), l_running_loss_train)
    plt.xlabel("# of epochs")
    plt.ylabel("total loss")
    plt.savefig("../output/{}_train_plot.png".format(file_name))



if __name__ == "__main__":
    file_name="data1_p"
    main(file_name)
    file_name="data2_p"
    main(file_name)
