import pandas as pd
import glob
from sklearn import metrics
import numpy as np

from scipy.optimize import fmin
from functools import partial
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler



def run_training(pred_df, fold):

    train_df = pred_df[pred_df.kfold != fold].reset_index(drop=True)
    valid_df = pred_df[pred_df.kfold == fold].reset_index(drop=True)

    xtrain = train_df[["lr_pred","lr_cnt_pred","rf_svd_pred"]].values
    xvalid = valid_df[["lr_pred","lr_cnt_pred","rf_svd_pred"]].values


    opt = LogisticRegression()
    
    opt.fit(xtrain, train_df.is_duplicate.values)
    preds = opt.predict_proba(xvalid)[:,1]
    auc = metrics.roc_auc_score(valid_df.is_duplicate.values, preds)
    print(f"{fold}, {auc}")
    valid_df.loc[:,"opt_pred"] =preds
    return opt.coef_

if __name__=="__main__":
    files = glob.glob("../model_preds/*.csv")
    df=None
    for f in files:
        if df is None:
            df = pd.read_csv(f)
        else:
            temp_df = pd.read_csv(f) 
            df = df.merge(temp_df, on='id',how='left')
    targets = df.is_duplicate.values
    pred_cols =["lr_pred","lr_cnt_pred","rf_svd_pred"]
    preds_df =[]
    coefs=[]
    for j in range(5):
        coefs.append(run_training(df,j))

    coefs=np.array(coefs)
    print(coefs)
    coefs = np.mean(coefs, axis=0)
    print(coefs)
    wt_avg=(
        coefs[0][0]*df.lr_pred
        +coefs[0][1]*df.lr_cnt_pred
        +coefs[0][2]*df.rf_svd_pred
    )
    print("optical auc after finding coefs")
    print(metrics.roc_auc_score(targets,wt_avg))