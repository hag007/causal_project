import sklearn
from sklearn.model_selection import train_test_split
import pandas as pd
import os
def split_train_test(file_name, test_size=0.2):
    path_data = "/media/hag007/Data/repos/hw4_dl/datasets/"
    bg_genes = open(os.path.join(path_data, "bg_genes.txt")).read().split("\n")
    affecting_genes = open(os.path.join(path_data, "affecting_genes.txt")).read().split("\n")
    if affecting_genes[0] == '':
        affecting_genes = []
    outcome_genes = open(os.path.join(path_data, "outcome_genes.txt")).read().split("\n")
    df = pd.read_csv("../datasets/{}.csv".format(file_name), index_col=0).loc[:,bg_genes+affecting_genes+outcome_genes+['T']]
    df[df["T"] == 1].to_csv("../datasets/{}_treated.csv".format(file_name))
    df[df["T"] == 0].to_csv("../datasets/{}_control.csv".format(file_name))

if __name__=="__main__":
    file_name="data1_p"
    split_train_test(file_name)

