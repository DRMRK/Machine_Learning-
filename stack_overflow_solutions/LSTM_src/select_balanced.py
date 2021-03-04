"""
Select tiny part of data

    # reads the whole datafile
    # creates a tiny subset of data with 
    same traget distribution as the original data

"""
import pandas as pd
from sklearn.model_selection import train_test_split

from imblearn.ensemble import BalancedBaggingClassifier
from imblearn.over_sampling import SMOTE, ADASYN 
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline 
from imblearn.over_sampling import KMeansSMOTE


if __name__=="__main__":
    
    df = pd.read_csv('../input/raw_data/train.csv')
    # add a id column and set index values
    df ["id"] =df.PostId
    # take a small subset of data

    df ["question"] = df["Title"] + ' ' + df["BodyMarkdown"]
    df.OpenStatus = df.OpenStatus.apply(
        lambda x:1 if x=="open" else 0
        )

    Xdata = df[["id","question","OpenStatus"]]
    ydata = df[["OpenStatus"]]
    
    #over = RandomOverSampler(sampling_strategy=0.01)
    under = RandomUnderSampler(sampling_strategy=1)
        
    steps=[('u',under)]
        
    pipeline=Pipeline(steps=steps)
        #transform the datset    
    X_res, y_res = pipeline.fit_resample(Xdata, ydata)
    print(X_res.shape)
    print(X_res.OpenStatus.value_counts())
    X_res.to_csv("../LSTM_input/balanced_train.csv",index=False)
    print("data shape",X_res.shape)
    print("done")
