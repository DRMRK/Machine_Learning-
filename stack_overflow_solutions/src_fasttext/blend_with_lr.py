import pandas as pd
import glob
from sklearn import metrics
import numpy as np

from scipy.optimize import fmin
from functools import partial
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import xgboost as xgb 
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix

from sklearn import linear_model



def run_training(pred_df, fold):

    train_df = pred_df[pred_df.kfold != fold].reset_index(drop=True)
    valid_df = pred_df[pred_df.kfold == fold].reset_index(drop=True)

  #  xtrain = train_df[["GNB_pred","lr_bagging_pred","lr_pred","xgb_pred_p","xgb_pred_n"]].values
   # xvalid = valid_df[["GNB_pred","lr_bagging_pred","lr_pred","xgb_pred_p","xgb_pred_n"]].values

    xtrain = train_df[["GNB_pred","lr_pred","xgb_pred_p","xgb_pred_n"]].values
    xvalid = valid_df[["GNB_pred","lr_pred","xgb_pred_p","xgb_pred_n"]].values
    


    le = preprocessing.LabelEncoder()
    le.fit(train_df.target)
    #print(le.classes_)
    ytrain = le.transform(train_df.target)

    yvalid = le.transform(valid_df.target)

    
        
    
    clf=linear_model.LogisticRegression(
        penalty ='l2',
        C=0.05,
        max_iter=5000,
        class_weight='balanced',
        solver='liblinear'
        )
        
    clf.fit(xtrain, ytrain)
    preds = clf.predict_proba(xvalid)[:,1]
    pred=clf.predict(xvalid)
    auc = metrics.roc_auc_score(yvalid, preds)
    print(f"{fold}, {auc}")
    valid_df.loc[:,"xgb_pred"] =preds
    conf_m=confusion_matrix(yvalid,pred)
    print(conf_m)
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
    targets = df.target.values
    pred_cols =["GNB_pred","lr_pred","xgb_pred_p","xgb_pred_n"]
    
    dfs =[]
    for j in range(5):
        temp_df = run_training(df,j)
        dfs.append(temp_df)
    fin_valid_df = pd.concat(dfs)
    print(
        metrics.roc_auc_score(
            fin_valid_df.target,fin_valid_df.xgb_pred.values
            )
        )