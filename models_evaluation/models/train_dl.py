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
import os



class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.input_layer=nn.Linear(233,160)
        self.h1=nn.Linear(160,80)
        self.h2 = nn.Linear(80, 40)
        self.h3 = nn.Linear(40, 20)
        self.h4=nn.Linear(20,10)
        self.out_layer=nn.Linear(10,1)

    def learn(self, x):
        h=self.input_layer(x)
        h=self.h1(F.relu(h))
        h=self.h2(F.relu(h))
        h = self.h3(F.relu(h))
        h = self.h4(F.relu(h))
        h=self.out_layer(F.relu(h))

        return h


class MyDataset(Dataset):
    def __init__(self, file_name, samples_type, pseudo=None):
        bg_genes = open(os.path.join("../datasets/", "bg_genes.txt")).read().split("\n")
        affecting_genes = open(os.path.join("../datasets/", "affecting_genes.txt")).read().split("\n")
        outcome_genes = open(os.path.join("../datasets/", "outcome_genes.txt")).read().split("\n")
        if affecting_genes[0] == '':
            affecting_genes = []
        fs = bg_genes + affecting_genes + ['T']
        self.samples=pd.read_csv("../datasets/{}_X_{}.csv".format(file_name, samples_type), index_col=0).loc[:, fs] # .drop(['Y'], axis=1)

        self.labels=pd.read_csv("../datasets/{}_y_{}.csv".format(file_name, samples_type), index_col=0).loc[:, outcome_genes] # .loc[:,'Y']
        self.samples=tensor(self.samples.values.astype(np.float)).float()
        self.labels=tensor(self.labels.values.astype(np.float)).float()
    def __len__(self):
        return self.samples.shape[0]

    def __getitem__(self, idx):
        return self.samples[idx], self.labels[idx]


def main(file_name):
    MOD_FACTOR=10 ** 0
    print "is CUDA available? ", torch.cuda.is_available()
    torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    dataset_train=MyDataset(file_name, "train")
    dataset_test=MyDataset(file_name, "test")

    loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=10,  shuffle=True, pin_memory=False)
    loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=dataset_test.__len__(),  shuffle=True, pin_memory=False)
    net=Net()
    lr = 0.0001
    optimizer = optim.Adam(net.parameters(), lr=lr)
    start=time.time()
    l_running_loss_train=[]
    l_running_loss_test=[]
    n_epochs=200
    for cur_epoch in np.arange(n_epochs):

        lr = 0.0001 if cur_epoch < n_epochs/2 else 0.00001
        optimizer = optim.Adam(net.parameters(), lr=lr)

        running_loss_train=0
        running_loss_test = 0
        for i, data in enumerate(loader_train, 0):
            x, y = data
            y_hat=net.learn(x)
            optimizer.zero_grad()
            loss = torch.nn.MSELoss()(y_hat, y)
            loss.backward()
            optimizer.step()
            running_loss_train+=loss.item()
        running_loss_train /= (i + 1)


        if cur_epoch%MOD_FACTOR==0:
            with torch.no_grad():
                for i, data in enumerate(loader_test, 0):
                    x, y = data
                    y_hat = net.learn(x)
                    test_loss = nn.MSELoss()(y_hat, y)
                    running_loss_test += test_loss.item()

                running_loss_test /= (i + 1)
                l_running_loss_test.append(np.sqrt(running_loss_test))

        l_running_loss_train.append(np.sqrt(running_loss_train))



        if cur_epoch %MOD_FACTOR == 0:
            print "epoch#{}: loss train: {}".format(cur_epoch, np.sqrt(running_loss_train))
            print "epoch#{}: loss test: {}".format(cur_epoch, np.sqrt(running_loss_test))
            print ""



            plt.plot(np.arange(1, len(l_running_loss_train)+1), l_running_loss_train)
            plt.xlabel("# of epochs")
            plt.ylabel("total loss")
            plt.savefig("../output/{}_train_plot.png".format(file_name))

            plt.plot(np.arange(1, len(l_running_loss_test)+1), l_running_loss_test)
            plt.xlabel("# of epochs")
            plt.ylabel("total loss")
            plt.savefig("../output/{}_test_plot.png".format(file_name))

    end = time.time()
    torch.save(net.state_dict(), "../output/dl_model_{}".format(file_name))
    print "total time: {}".format(end - start)


if __name__ == "__main__":
    file_name="data1_p"
    main(file_name)