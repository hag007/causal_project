import os
import pandas as pd
import numpy as np
import torch
from torch import tensor
import torch.optim as optim
from torch.utils.data import Dataset
from models.train_dl import Net
import time

class MyDataset(Dataset):
    def __init__(self, file_name, samples_type, treatment_value):
        bg_genes = open(os.path.join("../datasets/", "bg_genes.txt")).read().split("\n")
        affecting_genes = open(os.path.join("../datasets/", "affecting_genes.txt")).read().split("\n")
        outcome_genes = open(os.path.join("../datasets/", "outcome_genes.txt")).read().split("\n")
        if affecting_genes[0] == '':
            affecting_genes = []
        fs = bg_genes + affecting_genes + ['T']
        self.samples=pd.read_csv("../datasets/{}{}.csv".format(file_name, samples_type), index_col=0).loc[:, fs]
        self.labels=pd.read_csv("../datasets/{}{}.csv".format(file_name, samples_type),index_col=0).loc[:,outcome_genes]

        if treatment_value is not None:
            self.samples.loc[:,'T']=treatment_value

        self.samples=tensor(self.samples.values.astype(np.float)).float()
        self.labels=tensor(self.labels.values.astype(np.float)).float()
    def __len__(self):
        return self.samples.shape[0]

    def __getitem__(self, idx):
        return self.samples[idx], self.labels[idx]


def load_dl_model(file_name, samples_type):
    MOD_FACTOR=10 ** 1
    print "is CUDA available? ", torch.cuda.is_available()
    torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    dataset_train=MyDataset(file_name, samples_type, None)

    net=Net()
    if os.path.exists("../output/dl_model_s_learner_{}{}".format(file_name, samples_type)):
        net.load_state_dict(torch.load("../output/dl_model_s_learner_{}{}".format(file_name, samples_type)))
        net.eval()
        return net
    else:
        loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=10,  shuffle=True, pin_memory=False)
        start=time.time()
        l_running_loss_train=[]
        n_epochs=20
        cur_epoch=0
        lr = 0.0001 if cur_epoch < n_epochs/2 else 0.00001
        optimizer = optim.Adam(net.parameters(), lr=lr)
        for cur_epoch in np.arange(n_epochs):

            running_loss_train=0
            for i, data in enumerate(loader_train, 0):
                x, y = data
                y_hat=net.learn(x)
                optimizer.zero_grad()
                loss = torch.nn.MSELoss()(y_hat, y)
                loss.backward()
                optimizer.step()
                running_loss_train+=loss.item()
            running_loss_train /= (i + 1)

            l_running_loss_train.append(running_loss_train)

            if cur_epoch %MOD_FACTOR == 0:
                print "epoch#{}: loss train: {}".format(cur_epoch, running_loss_train)
                print ""

        end=time.time()
        torch.save(net.state_dict(), "../output/dl_model_s_learner_{}{}".format(file_name, samples_type))
        print "total time: {}".format(end-start)
        return net



def calc_s_learner(file_name, is_treatment_group=True):

    net=load_dl_model(file_name, "")

    group_type=("_treated" if is_treatment_group else "_control")
    dataset_treated_1=MyDataset(file_name,group_type, 1)
    loader_treated_1 = torch.utils.data.DataLoader(dataset_treated_1, batch_size=1,  shuffle=True, pin_memory=False)
    dataset_treated_0=MyDataset(file_name,group_type, 0)
    loader_treated_0 = torch.utils.data.DataLoader(dataset_treated_0, batch_size=1,  shuffle=True, pin_memory=False)
    with torch.no_grad():
        for i, data in enumerate(loader_treated_1, 0):
            x, y_train = data
            y_hat_treated = net.learn(x)
        for i, data in enumerate(loader_treated_0, 0):
            x, y_test = data
            y_hat_control = net.learn(x)

        att = torch.mean(y_hat_treated-y_hat_control).cpu().numpy()

        return att, dataset_treated_1.__len__()

if __name__=="__main__":

    att, n_att=calc_s_learner("data1_p", True)
    atc, n_atc=calc_s_learner("data1_p", False)
    n_samples=float(n_att + n_atc)

    try:
        df = pd.read_csv("../output/agg_result.csv", index_col=0)
    except:
        df = pd.DataFrame()

    df.loc[0, "atc"] = atc
    df.loc[0, "atc_fraction"] = n_atc / n_samples
    df.loc[0, "att"] = att
    df.loc[0, "att_fraction"] = n_att / n_samples
    df.loc[0, "ate"] = atc * (n_atc / n_samples) + att * (n_att / n_samples)

    df.to_csv("../output/agg_result.csv")