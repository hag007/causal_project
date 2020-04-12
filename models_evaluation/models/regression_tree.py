import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import numpy as np
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
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
    fs = bg_genes + affecting_genes + ['T']
    df_X_train = df_X_train.loc[:, fs]
    df_X_test = df_X_test.loc[:, fs]

    regressor = RandomForestRegressor(n_estimators = 100, random_state = 0)

    # fit the regressor with x and y data
    model=regressor.fit(df_X_train, df_y_train)

    y_hat=model.predict(df_X_test)
    loss=np.sqrt(mean_squared_error(y_hat, df_y_test))
    plt.clf()
    sns.distplot(y_hat, norm_hist=False, kde=False)
    plt.savefig("../output/dist_reg_tree_{}.png".format(file_name))
    file("../output/regression_tree_{}".format(file_name),'w+').write(pickle.dumps(model))
    print "loss_regression_tree: {}".format(loss)






if __name__=="__main__":

    loss_1 =main("data1_p")

