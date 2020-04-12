import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import numpy as np
import pickle
import seaborn as sns
import matplotlib.pyplot as plt

def main(file_name):
    df_X_train = pd.read_csv("../datasets/{}_X_train.csv".format(file_name), index_col=0).drop(['T', 'scores'], axis=1)
    df_y_train = pd.read_csv("../datasets/{}_X_train.csv".format(file_name), index_col=0).loc[:,['T']]
    df_X_test = pd.read_csv("../datasets/{}_X_test.csv".format(file_name), index_col=0).drop(['T', 'scores'], axis=1)
    df_y_test = pd.read_csv("../datasets/{}_X_test.csv".format(file_name), index_col=0).loc[:,['T']]

    regressor = RandomForestRegressor(n_estimators = 100, random_state = 0)

    # fit the regressor with x and y data
    model=regressor.fit(df_X_train, df_y_train)

    y_hat=model.predict(df_X_test)
    loss=mean_squared_error(y_hat, df_y_test)

    plt.clf()
    sns.distplot(y_hat, norm_hist=False, kde=False)
    plt.savefig("../output/dist_reg_tree_{}.png".format(file_name))


    file("../output/regression_tree_ps_{}".format(file_name),'w+').write(pickle.dumps(model))
    print "loss_regression_tree: {}".format(loss)






if __name__=="__main__":

    loss_1 =main("data1_p")
    print "========================"
    loss_2=main("data2_p")
