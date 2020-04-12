import pandas as pd


def calc_ipw(file_name):
    df=pd.read_csv("../datasets/{}.csv".format(file_name), index_col=0)
    df_treated=df[df["T"]==1]
    df_control = df[df["T"] == 0]

    treated_expr=df_treated["Y"].sum()/len(df_treated.index)
    control_expr = df_control.apply(lambda row: row["Y"]*row["scores"]*(1-row["scores"]),axis=1).sum()/len(df_control.index)

    att=treated_expr-control_expr
    print att
    return att

if __name__=="__main__":

    att_1=calc_ipw("data1_p")
    att_2=calc_ipw("data2_p")
    try:
        df=pd.read_csv("../output/agg_result.csv",index_col=0)
    except:
        df =pd.DataFrame()

    df.loc[1,"data1"]=att_1
    df.loc[1, "data2"]=att_2
    df.to_csv("../output/agg_result.csv")