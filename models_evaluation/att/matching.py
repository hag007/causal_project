import pandas as pd
from sklearn.neighbors import NearestNeighbors
import numpy as np
from scipy.stats import zscore
import os

def calc_matching(file_name, is_treatment=1):
    topn=10
    n_jobs=10

    df=pd.read_csv("../datasets/{}.csv".format(file_name), index_col=0)

    df_treated=df[df["T"]==is_treatment]
    df_control = df[df["T"] == np.abs(1-is_treatment)]

    bg_genes = open(os.path.join("../datasets/", "bg_genes.txt")).read().split("\n")
    affecting_genes = open(os.path.join("../datasets/", "affecting_genes.txt")).read().split("\n")
    outcome_genes = open(os.path.join("../datasets/", "outcome_genes.txt")).read().split("\n")
    if affecting_genes[0] == '':
        affecting_genes = []
    fs = bg_genes + affecting_genes + ['T']
    df_1=df.loc[:,fs+outcome_genes]
    df_1=pd.concat([df_1.drop(["T"],axis=1), df_1.loc[:,["T"]]], axis=1)
    df_1_treated=df_1[df_1["T"]==1]
    df_1_control = df_1[df_1["T"] == 0]
    indices_1=get_neighbors(df_1_treated, df_1_control, 'euclidean', outcome_genes, topn=10)

    y_1_hats=[]
    for cur_1 in zip(indices_1.tolist()):
        vfunc = np.vectorize(lambda a: df_control.iloc[a].loc[outcome_genes])
        indices= cur_1

        y_1_hats.append(vfunc(list(cur_1)).mean())
        vfunc = np.vectorize(lambda a: df_control.iloc[a].loc[outcome_genes])

    print  "test:"
    print (df_treated.loc[:,outcome_genes].values-np.array(y_1_hats)).mean()


    ite=df_treated.loc[:,outcome_genes].values-y_1_hats
    att=ite.mean()

    print att, len(ite)
    return att, len(ite)


def get_neighbors(df_treated, df_control, metric, outcome_genes, topn=20, n_jobs=10, params={}):
    X_t=df_treated.drop(["T"]+outcome_genes, axis=1)
    X_c=df_control.drop(["T"]+outcome_genes, axis=1)

    knn = NearestNeighbors(algorithm='brute', n_neighbors=topn, metric=metric, n_jobs=n_jobs, metric_params=params) #
    knn.fit(X_c)
    distances, indices = knn.kneighbors(X_t)

    return indices


if __name__=="__main__":


    att, n_att=calc_matching("data1_p", True)
    atc, n_atc=calc_matching("data1_p", False)
    n_samples=float(n_att + n_atc)

    try:
        df = pd.read_csv("../output/agg_result.csv", index_col=0)
    except:
        df = pd.DataFrame()

    df.loc[4, "atc"] = atc
    df.loc[4, "atc_fraction"] = n_atc / n_samples
    df.loc[4, "att"] = att
    # df.loc[4, "att_fraction"] = n_att / n_samples
    # df.loc[4, "ate"] = atc * (n_atc / n_samples) + att * (n_att / n_samples)

    df.to_csv("../output/agg_result.csv")

