import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import os

def calc_t_learner(file_name, is_treatment_group=True):
    group_type = (1 if is_treatment_group else 0)
    bg_genes = open(os.path.join("../datasets/", "bg_genes.txt")).read().split("\n")
    affecting_genes = open(os.path.join("../datasets/", "affecting_genes.txt")).read().split("\n")
    outcome_genes = open(os.path.join("../datasets/", "outcome_genes.txt")).read().split("\n")
    if affecting_genes[0] == '':
        affecting_genes = []
    fs = bg_genes + affecting_genes + ['T']
    df = pd.read_csv("../datasets/{}.csv".format(file_name), index_col=0)
    df = df.loc[:, outcome_genes + fs]
    df_treated=df[df["T"]==group_type]
    df_control = df[df["T"] == np.abs(1-group_type)]

    model_1 = RandomForestRegressor()
    model_1.fit(df_treated.drop(outcome_genes+["T"],axis=1), df_treated[outcome_genes])
    loss_1 = mean_squared_error(df_treated.loc[:, outcome_genes], model_1.predict(df_treated.drop(outcome_genes+['T'] , axis=1)))
    # print "loss: {}".format(loss_1)
    res_1=model_1.predict(df_treated.drop(outcome_genes+["T"],axis=1))

    model_2 = RandomForestRegressor()
    model_2.fit(df_control.drop(outcome_genes+["T"], axis=1), df_control[outcome_genes])
    loss_2 = mean_squared_error(df_control.loc[:, outcome_genes], model_2.predict(df_control.drop(outcome_genes+['T'], axis=1)))
    # print "loss: {}".format(loss_2)
    res_2=model_2.predict(df_treated.drop(outcome_genes+ ["T"],axis=1))

    print "loss: {}".format((loss_1*len(df_treated.index)+ loss_2*len(df_treated.index))/len(df.index))

    att=(res_1-res_2).mean()
    print att
    return att, df_treated.shape[0]

if __name__=="__main__":



    att, n_att=calc_t_learner("data1_p", True)
    atc, n_atc=calc_t_learner("data1_p", False)
    n_samples=float(n_att + n_atc)

    try:
        df=pd.read_csv("../output/agg_result.csv",index_col=0)
    except:
        df =pd.DataFrame()
    df.loc[3, "atc"] = atc
    df.loc[3, "atc_fraction"] = n_atc / n_samples
    df.loc[3, "att"] = att
    df.loc[3, "att_fraction"] = n_att / n_samples
    df.loc[3, "ate"] = atc * (n_atc / n_samples) + att * (n_att / n_samples)

    df.to_csv("../output/agg_result.csv")