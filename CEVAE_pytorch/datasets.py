import numpy as np
from sklearn.model_selection import train_test_split
import torch
import pandas as pd
import os
from torch.utils.data.sampler import SubsetRandomSampler

class IHDP(object):
    def __init__(self, path_data="datasets/IHDP/", replications=10):
        self.path_data = path_data
        self.replications = replications
        # which features are binary
        self.binfeats = [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]
        # which features are continuous
        self.contfeats = [i for i in range(25) if i not in self.binfeats]

    def __iter__(self):
        for i in range(self.replications):
            data = np.loadtxt(self.path_data + '/ihdp_npci_' + str(i + 1) + '.csv', delimiter=',')
            t, y, y_cf = data[:, 0], data[:, 1][:, np.newaxis], data[:, 2][:, np.newaxis]
            mu_0, mu_1, x = data[:, 3][:, np.newaxis], data[:, 4][:, np.newaxis], data[:, 5:]
            yield (x, t, y), (y_cf, mu_0, mu_1)

    def get_train_valid_test(self):
        for i in range(self.replications):
            data = np.loadtxt(self.path_data + '/ihdp_npci_' + str(i + 1) + '.csv', delimiter=',')
            t, y, y_cf = data[:, 0][:, np.newaxis], data[:, 1][:, np.newaxis], data[:, 2][:, np.newaxis]
            mu_0, mu_1, x = data[:, 3][:, np.newaxis], data[:, 4][:, np.newaxis], data[:, 5:]
            # this binary feature is in {1, 2}
            x[:, 13] -= 1
            idxtrain, ite = train_test_split(np.arange(x.shape[0]), test_size=0.2, random_state=1)
            itr, iva = train_test_split(idxtrain, test_size=0.3, random_state=1)
            train = (x[itr], t[itr], y[itr]), (y_cf[itr], mu_0[itr], mu_1[itr])
            valid = (x[iva], t[iva], y[iva]), (y_cf[iva], mu_0[iva], mu_1[iva])
            test = (x[ite], t[ite], y[ite]), (y_cf[ite], mu_0[ite], mu_1[ite])
            yield train, valid, test, self.contfeats, self.binfeats

#########################################################################################################

class CMAP(object):
    def __init__(self, path_data="datasets/CMAP/", data_case="/media/hag007/Data/cmap_datasets/profiles_bortezomib_6290_level_4_GSE92742_GSE70138.tsv", data_control="/media/hag007/Data/cmap_datasets/profiles_EMPTY_VECTOR_5114_level_4_GSE92742_GSE70138.tsv", replications=1):
        self.path_data = path_data
        self.data_case=data_case
        self.data_control=data_control
        self.replications = replications
        # which features are binary

        self.bg_genes = open(os.path.join(self.path_data, "bg_genes.txt")).read().split("\n")
        self.affecting_genes = open(os.path.join(self.path_data, "affecting_genes.txt")).read().split("\n")
        if self.affecting_genes[0]=='':
            self.affecting_genes=[]
        self.outcome_genes = open(os.path.join(self.path_data, "outcome_genes.txt")).read().split("\n")
        self.binfeats = np.arange(len(self.affecting_genes))

        # which features are continuous
        self.contfeats = [i for i in range(len(self.affecting_genes), len(self.affecting_genes)+len(self.bg_genes))]

    def __iter__(self):
        for i in range(self.replications):
            df_data_case = pd.read_csv(self.data_case, sep='\t', index_col=0).drop(["pr_gene_symbol", "pr_gene_symbol.1"],
                                                                              axis=1).T
            df_data_case.T = 1
            df_data_control = pd.read_csv(self.data_control, sep='\t', index_col=0).drop(
                ["pr_gene_symbol", "pr_gene_symbol.1"], axis=1).T
            df_data_control.T = 0
            df_data = pd.concat([df_data_case, df_data_control], axis=0)



            t, y, = df_data.T, df_data.loc[:, self.outcome_genes[0]]
            x =  df_data.loc[:, self.affecting_genes+self.bg_genes]
            yield (x, t, y)

    def get_train_valid_test(self):
        for i in range(self.replications):

            df_data_case = pd.read_csv(self.data_case, sep='\t', index_col=0).drop(
                ["pr_gene_symbol", "pr_gene_symbol.1"],
                axis=1).T
            df_data_case.T = 1
            df_data_control = pd.read_csv(self.data_control, sep='\t', index_col=0).drop(
                ["pr_gene_symbol", "pr_gene_symbol.1"], axis=1).T
            df_data_control.T = 0
            df_data = pd.concat([df_data_case, df_data_control], axis=0)

            bg_genes = open(os.path.join(self.path_data, "bg_genes.txt")).read().split("\n")
            affecting_genes = open(os.path.join(self.path_data, "affecting_genes.txt")).read().split("\n")
            if affecting_genes[0] == '':
                affecting_genes = []
            outcome_genes = open(os.path.join(self.path_data, "outcome_genes.txt")).read().split("\n")

            t, y, = df_data.loc[:, ['T']], df_data.loc[:, outcome_genes]
            x = df_data.loc[:, affecting_genes+bg_genes]

            idxtrain, ite = train_test_split(np.arange(x.shape[0]), test_size=0.1, random_state=1)
            itr, iva = train_test_split(idxtrain, test_size=0.3, random_state=1)
            train = (x.iloc[itr].values, t.iloc[itr].values, y.iloc[itr].values)
            valid = (x.iloc[iva].values, t.iloc[iva].values, y.iloc[iva].values)
            test = (x.iloc[ite].values, t.iloc[ite].values, y.iloc[ite].values)
            yield train, valid, test, self.contfeats, self.binfeats

    ###########################################################################################

class CMAPDataset(torch.utils.data.Dataset):
    def __init__(self, data_case, data_control):
        df_data_case = pd.read_csv(data_case, sep='\t', index_col=0).drop(["pr_gene_symbol", "pr_gene_symbol.1"],
                                                                          axis=1).T
        df_data_case.T = 1
        df_data_control = pd.read_csv(data_control, sep='\t', index_col=0).drop(
            ["pr_gene_symbol", "pr_gene_symbol.1"], axis=1).T
        df_data_control.T = 0
        df_data = pd.concat([df_data_case, df_data_control], axis=0)

        path = os.path.dirname(os.path.abspath(__file__))
        bg_genes = open(os.path.join(path, "data/CMAP/bg_genes.txt")).read().split("\n")
        affecting_genes = [] # open(os.path.join(path, "data/CMAP/affecting_genes.txt")).read().split("\n")
        outcome_genes = open(os.path.join(path, "data/CMAP/outcome_genes.txt")).read().split("\n")

        self.x_bg = torch.Tensor(df_data.loc[:, bg_genes].values).double()
        self.x_af = torch.Tensor(df_data.loc[:, affecting_genes].values).double()
        self.x = torch.Tensor(df_data.loc[:, bg_genes + affecting_genes].values).double()
        self.y = torch.Tensor(df_data.loc[:, outcome_genes].values).double()
        self.t = torch.Tensor(df_data.loc[:, 'T'].values).double()

        self.length = df_data.shape[0]

        # Zero mean, unit variance for y during training
        self.y_mean, self.y_std = self.y.mean(), self.y.std()
        self.standard_yf = (self.y - self.y_mean) / self.y_std
        # self.standard_yf=torch.Tensor(self.standard_yf.values)

    def __getitem__(self, index):
        return self.t[index], self.x[index], self.y[index], self.standard_yf[index]

    def __len__(self):
        return self.length

    def indices_bg_af(self):
        return self.x_bg, self.x_af

    def y_mean_std(self):
        return self.y_mean, self.y_std

class CMAPDataLoader(object):
    def __init__(self, dataset, validation_split, shuffle=True):
        dataset_size = len(dataset)
        indices = list(range(dataset_size))
        split = int(np.floor(validation_split * dataset_size))
        if shuffle:
            np.random.seed(42)
            np.random.shuffle(indices)
        train_indices, valid_indices = indices[split:], indices[: split]

        self.dataset = dataset
        self.train_sampler = SubsetRandomSampler(train_indices)
        self.valid_sampler = SubsetRandomSampler(valid_indices)

    def train_loader(self, batch_size):
        train_loader = torch.utils.data.DataLoader(
            dataset=self.dataset, batch_size=batch_size, sampler=self.train_sampler)

        return train_loader

    def test_loader(self, batch_size):
        test_loader = torch.utils.data.DataLoader(
            dataset=self.dataset, batch_size=batch_size, sampler=self.valid_sampler)

        return test_loader

    def loaders(self, batch_size):
        train_loader = self.train_loader(batch_size)
        test_loader = self.test_loader(batch_size)

        return train_loader, test_loader
