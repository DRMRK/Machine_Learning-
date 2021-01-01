import pandas as pd
import glob
from sklearn import metrics
import numpy as np

if __name__=="__main__":
    
    files = glob.glob("../model_preds/*.csv")
    df=None
    for f in files:
        if df is None:
            df = pd.read_csv(f)
        else:
            temp_df = pd.read_csv(f) 
            df = df.merge(temp_df, on='id',how='left')

    #print(df.head(10))    
    targets = df.is_duplicate.values
    pred_cols=["lr_pred","lr_cnt_pred","rf_svd_pred"]

    for col in pred_cols:
        auc = metrics.roc_auc_score(df.is_duplicate.values, df[col].values)
        print(f"pred_col {col}, ovearall_auc ={auc}")

    print('average')
    avg_pred =np.mean(df[["lr_pred","lr_cnt_pred","rf_svd_pred"]].values,axis=1)
    print(metrics.roc_auc_score(targets,avg_pred))

    print("Weighted average")
    lr_pred = df.lr_pred.values
    lr_cnt_pred = df.lr_cnt_pred.values
    rf_svd_pred = df.rf_svd_pred.values
    avg_pred = (3*rf_svd_pred+lr_cnt_pred+lr_pred)/5
    print(metrics.roc_auc_score(targets,avg_pred))

    print("rank averaging")
    lr_pred = df.lr_pred.rank().values
    lr_cnt_pred = df.lr_cnt_pred.rank().values
    rf_svd_pred = df.rf_svd_pred.rank().values
    avg_pred = (rf_svd_pred+lr_cnt_pred+lr_pred)/3
    print(metrics.roc_auc_score(targets,avg_pred))


    print("weighted rank averaging")
    lr_pred = df.lr_pred.rank().values
    lr_cnt_pred = df.lr_cnt_pred.rank().values
    rf_svd_pred = df.rf_svd_pred.rank().values
    avg_pred = (3*rf_svd_pred+lr_cnt_pred+lr_pred)/5
    print(metrics.roc_auc_score(targets,avg_pred))