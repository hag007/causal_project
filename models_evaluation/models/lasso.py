import pandas as pd
from sklearn.linear_model import LinearRegression, ElasticNetCV, RidgeCV, LassoCV, BayesianRidge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np
import pickle
import os

def main(file_name):
    df_X_train = pd.read_csv("../datasets/{}_X_train.csv".format(file_name), index_col=0)
    df_X_test = pd.read_csv("../datasets/{}_X_test.csv".format(file_name), index_col=0)
    df_y_train = pd.read_csv("../datasets/{}_y_train.csv".format(file_name), index_col=0)
    df_y_test = pd.read_csv("../datasets/{}_y_test.csv".format(file_name), index_col=0, )
    bg_genes = open(os.path.join("../datasets/", "bg_genes.txt")).read().split("\n")
    affecting_genes = open(os.path.join("../datasets/", "affecting_genes.txt")).read().split("\n")
    if affecting_genes[0] == '':
        affecting_genes = []
    fs= bg_genes+affecting_genes+['T']
    df_X_train=df_X_train.loc[:,fs]
    df_X_test=df_X_test.loc[:,fs]


    model=LassoCV().fit(df_X_train, df_y_train)
    y_hat=model.predict(df_X_test)
    loss=np.sqrt(mean_squared_error(y_hat, df_y_test))
    file("../output/lasso_{}".format(file_name),'w+').write(pickle.dumps(model))
    print "loss_lasso: {}".format(loss)






if __name__=="__main__":

    loss_1 =main("data1_p")
