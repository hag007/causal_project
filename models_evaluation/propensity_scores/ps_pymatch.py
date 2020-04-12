import warnings
warnings.filterwarnings('ignore')
from pymatch.Matcher import Matcher
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt


fields = \
[
    "x_1",
    "x_2",
    "x_3",
    "x_4",
    "x_5",
    "x_6",
    "x_7",
    "x_8",
    "x_9",
    "x_10",
    "x_11",
    "x_12",
    "x_13",
    "x_14",
    "x_15",
    "x_16",
    "x_17",
    "x_18",
    "x_19",
    "x_20",
    "x_21",
    "x_22",
    "x_23",
    "x_24",
    "x_25",
    "x_26",
    "x_27",
    "x_28",
    "x_29",
    "x_30",
    "x_31",
    "x_32",
    "x_33",
    "x_34",
    "x_35",
    "x_36",
    "x_37",
    "x_38",
    "x_39",
    "x_40",
    "x_41",
    "x_42",
    "x_43",
    "x_44",
    "x_45",
    "x_46",
    "x_47",
    "x_48",
    "x_49",
    "x_50",
    "x_51",
    "x_52",
    "x_53",
    "x_54",
    "x_55",
    "x_56",
    "x_57",
    "x_58",
    "T",
    "Y"
]


# fields = ['x_18', 'x_57', 'x_30', 'x_41', 'x_58', 'x_6', 'x_4', 'x_5', 'x_3', 'x_36', 'x_37', 'x_23', 'x_27',
#          'x_26', 'x_29', 'x_28', 'x_38', 'T']

def calc_propensity_scores(file_name):
    data = pd.read_csv("../datasets/{}.csv".format(file_name), index_col=0)[fields]
    categorical_c=[]
    for a in data.columns:
        try:
            float(data.iloc[0].loc[a])
        except:
            categorical_c.append(a)

    print categorical_c
    data_dummy=pd.get_dummies(data, columns=categorical_c, drop_first=True)

    control=data_dummy[data_dummy["T"]==0]
    test=data_dummy[data_dummy["T"]==1]

    m = Matcher(test, control, yvar="T", exclude=["Y"])
    np.random.seed(20170925)
    m.fit_scores(balance=False, nmodels=1)
    m.predict_scores()
    m.plot_scores()
    plt.savefig("../output/pm_results_{}.png".format(file_name))
    m.data.to_csv("../datasets/{}_p.csv".format(file_name))
    return m.data["scores"]
if __name__=="__main__":
    file_name="data1"
    res_1=calc_propensity_scores(file_name)
    plt.clf()
    file_name = "data2"
    res_2=calc_propensity_scores(file_name)



    res=pd.concat([res_1, res_2], axis=1)
    res=res.T
    res.index=["data1", "data2"]
    res.to_csv("../output/propensity_scores.csv")
