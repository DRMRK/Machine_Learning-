import pandas as pd
import glob
from sklearn import metrics
import numpy as np

from scipy.optimize import fmin
from functools import partial
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import xgboost as xgb 



def run_training(pred_df, fold):

    train_df = pred_df[pred_df.kfold != fold].reset_index(drop=True)
    valid_df = pred_df[pred_df.kfold == fold].reset_index(drop=True)

    xtrain = train_df[["lr_pred","lr_cnt_pred","rf_svd_pred","gnb_pred"]].values
    xvalid = valid_df[["lr_pred","lr_cnt_pred","rf_svd_pred","gnb_pred"]].values


    clf = xgb.XGBRFClassifier(
        use_label_encoder=False,base_score=0.5, colsample_bylevel=1, colsample_bytree=1,
        gamma=0, learning_rate=0.1, max_delta_step=0, max_depth=10,
        min_child_weight=1, missing=None, n_estimators=100, nthread=-1,
        objective='binary:logistic',eval_metric='logloss'
        )
    
    clf.fit(xtrain, train_df.is_duplicate.values)
    preds = clf.predict_proba(xvalid)[:,1]
    auc = metrics.roc_auc_score(valid_df.is_duplicate.values, preds)
    print(f"{fold}, {auc}")
    valid_df.loc[:,"xgb_pred"] =preds
    return valid_df

if __name__=="__main__":
    files = glob.glob("../model_preds/*.csv")
    df=None
    for f in files:
        if df is None:
            df = pd.read_csv(f)
        else:
            temp_df = pd.read_csv(f) 
            df = df.merge(temp_df, on='id',how='left',suffixes=(None, '_y'))     
    targets = df.is_duplicate.values
    pred_cols =["lr_pred","lr_cnt_pred","rf_svd_pred","gnb_pred"]
    
    dfs =[]
    for j in range(5):
        temp_df = run_training(df,j)
        dfs.append(temp_df)
    fin_valid_df = pd.concat(dfs)
    print(
        metrics.roc_auc_score(
            fin_valid_df.is_duplicate,fin_valid_df.xgb_pred.values
            )
        )