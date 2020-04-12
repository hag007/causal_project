import sklearn
from sklearn.model_selection import train_test_split
import pandas as pd
import os

def split_train_test(file_name, test_size=0.2):
    path_data="/media/hag007/Data/repos/hw4_dl/datasets/"
    bg_genes = open(os.path.join(path_data, "bg_genes.txt")).read().split("\n")
    affecting_genes = open(os.path.join(path_data, "affecting_genes.txt")).read().split("\n")
    if affecting_genes[0] == '':
        affecting_genes = []
    outcome_genes = open(os.path.join(path_data, "outcome_genes.txt")).read().split("\n")
    df = pd.read_csv("../datasets/{}.csv".format(file_name), index_col=0).loc[:,bg_genes+affecting_genes+outcome_genes+['T']]
    X_train, X_test, y_train, y_test = train_test_split(df.drop(outcome_genes, axis=1), df.loc[:, outcome_genes[0]], test_size=test_size)
    X_train.to_csv("../datasets/{}_X_train.csv".format(file_name))
    X_test.to_csv("../datasets/{}_X_test.csv".format(file_name))
    y_train.to_frame().to_csv("../datasets/{}_y_train.csv".format(file_name), columns=outcome_genes)
    y_test.to_frame().to_csv("../datasets/{}_y_test.csv".format(file_name), columns=outcome_genes)

if __name__=="__main__":
    file_name="data1_p"
    split_train_test(file_name)

