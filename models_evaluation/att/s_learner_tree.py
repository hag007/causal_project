import pandas as pd
from sklearn.linear_model import LinearRegression, ElasticNetCV, RidgeCV, LassoCV, BayesianRidge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import os

def calc_s_learner(file_name, is_treatment_group=True):
    group_type = (1 if is_treatment_group else 0)
    bg_genes = open(os.path.join("../datasets/", "bg_genes.txt")).read().split("\n")
    affecting_genes = open(os.path.join("../datasets/", "affecting_genes.txt")).read().split("\n")
    outcome_genes = open(os.path.join("../datasets/", "outcome_genes.txt")).read().split("\n")
    if affecting_genes[0] == '':
        affecting_genes = []
    fs = bg_genes + affecting_genes + ['T']
    df = pd.read_csv("../datasets/{}.csv".format(file_name), index_col=0).loc[:,fs+ outcome_genes]
    df_group = df[df["T"] == group_type]

    model=RandomForestRegressor(n_estimators = 100, random_state = 0).fit(df.drop(outcome_genes, axis=1), df.loc[:,outcome_genes])
    loss=mean_squared_error(df.loc[:,outcome_genes], model.predict(df.drop(outcome_genes, axis=1)))
    print "loss: {}".format(loss)
    return calc_att(model, df_group.drop(outcome_genes, axis=1)), df_group.shape[0]


def calc_att(model, sample):
    sample.loc[:, "T"] = 1
    res_1 = model.predict(sample)
    sample.loc[:, "T"] = 0
    res_2 = model.predict(sample)
    att = (res_1 - res_2).mean()
    print "att: {}".format(att)
    return att


if __name__=="__main__":

    att, n_att=calc_s_learner("data1_p", True)
    atc, n_atc=calc_s_learner("data1_p", False)
    n_samples=float(n_att + n_atc)

    try:
        df = pd.read_csv("../output/agg_result.csv", index_col=0)
    except:
        df = pd.DataFrame()

    df.loc[2, "atc"] = atc
    df.loc[2, "atc_fraction"] = n_atc / n_samples
    df.loc[2, "att"] = att
    df.loc[2, "att_fraction"] = n_att / n_samples
    df.loc[2, "ate"] = atc * (n_atc / n_samples) + att * (n_att / n_samples)

    df.to_csv("../output/agg_result.csv")