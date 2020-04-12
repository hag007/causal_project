import torch
from train_dl import MyDataset, Net
import pandas as pd
import pickle
from sklearn.metrics import mean_squared_error
import numpy as np

def main(file_name):
    losses={}
    df_X_train = pd.read_csv("../datasets/{}_X_train.csv".format(file_name), index_col=0)
    df_X_test = pd.read_csv("../datasets/{}_X_test.csv".format(file_name), index_col=0)
    df_y_train = pd.read_csv("../datasets/{}_y_train.csv".format(file_name), index_col=0)
    df_y_test = pd.read_csv("../datasets/{}_y_test.csv".format(file_name), index_col=0, )
    X_all=pd.concat([df_X_test])
    y_all = pd.concat([df_y_test])


    reg = pickle.loads(file("../output/linear_regression_{}".format(file_name)).read())
    y_hat=reg.predict(X_all)
    losses["lin_reg"]=np.sqrt(mean_squared_error(y_hat, y_all))

    reg = pickle.loads(file("../output/lasso_{}".format(file_name)).read())
    y_hat=reg.predict(X_all)
    losses["lasso"]=np.sqrt(mean_squared_error(y_hat, y_all))

    reg = pickle.loads(file("../output/ridge_{}".format(file_name)).read())
    y_hat=reg.predict(X_all)
    losses["ridge"]=np.sqrt(mean_squared_error(y_hat, y_all))

    reg = pickle.loads(file("../output/elastic_net_{}".format(file_name)).read())
    y_hat=reg.predict(X_all)
    losses["elastic_net"]=np.sqrt(mean_squared_error(y_hat, y_all))

    reg = pickle.loads(file("../output/regression_tree_{}".format(file_name)).read())
    y_hat=reg.predict(X_all)
    losses["loss_reg_tree"]=np.sqrt(mean_squared_error(y_hat, y_all))

    dataset_train=MyDataset(file_name, "train")
    dataset_test=MyDataset(file_name, "test")
    loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=1,  shuffle=True, pin_memory=False)
    loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=dataset_test.__len__(),  shuffle=True, pin_memory=False)

    net = Net()
    net.load_state_dict(torch.load("../output/dl_model_{}".format(file_name)))
    net.eval()
    with torch.no_grad():
        for i, data in enumerate(loader_test, 0):
            x, y_test = data
            y_hat_test = net.learn(x)

    losses["loss_dl"]=np.sqrt(mean_squared_error(y_hat_test, y_test))

    return losses


if __name__=="__main__":
    file_name="data1_p"
    df=pd.DataFrame()
    losses=main(file_name)
    for k,v in losses.iteritems():
        df.loc[k,file_name]=v

    df.to_csv("../output/model_eval_results.csv", sep='\t')
