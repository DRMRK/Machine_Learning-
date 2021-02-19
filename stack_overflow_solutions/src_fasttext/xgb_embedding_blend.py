import numpy as np
import pandas as pd

from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn import preprocessing
import matplotlib.pyplot as plt

import xgboost as xgb

import time

from imblearn.ensemble import BalancedBaggingClassifier
from imblearn.over_sampling import SMOTE, ADASYN 
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline 
from imblearn.over_sampling import KMeansSMOTE

from collections import Counter

def run_training(fold_):
    total_roc = []
    total_conf =[]
    
    t0=time.time()
    #df = pd.read_csv("../input/embedded_train_tiny_folds.csv")
    df = pd.read_hdf(
        path_or_buf="../input/tiny_data/full_data_folds.h5",
        key='dataset'
        )
    #print("tg\n",df.target.value_counts())   
    #print(" ") 
    t1=time.time()
    total_time = t1-t0
    print("time to read file",total_time)

    print(f"fold: {fold_}")
        
    t0=time.time()

    train_df = df[df.kfold != fold_].reset_index(drop = True)
    test_df = df[df.kfold == fold_].reset_index(drop = True)
    #    print("train shape\n", train_df.shape)
     #   print("test shape\n", test_df.shape)
        
        #features
    xtrain = train_df.drop(["kfold","target"],axis=1)
    xtest =  test_df.drop(["kfold","target"],axis=1)
        # Standard scaler
        
    #sc = StandardScaler()
    #sc.fit(xtrain)

    #xtrain = sc.transform(xtrain)
    #xtest = sc.transform(xtest)
        
        # target
        # First make the target binary
    train_df.target = train_df.target.apply(
        lambda x:'open' if x=='open' else 'closed'
        )

    test_df.target = test_df.target.apply(
        lambda x:'open' if x=='open' else 'closed'
        )    
    # Encode labels 
    le = preprocessing.LabelEncoder()
    le.fit(train_df.target)
    #print(le.classes_)
    ytrain = le.transform(train_df.target)

    ytest = le.transform(test_df.target)

    print("now do SMOTE")
        # defin pipeline
        #over = RandomOverSampler(
        #    sampling_strategy=0.032,
        #    random_state=0)
    over = SMOTE(sampling_strategy=0.8, n_jobs=-1)
    under = RandomUnderSampler(
        sampling_strategy=0.9)
        
    steps=[('o',over),('u',under)]
        
    pipeline=Pipeline(steps=steps)
        #transform the datset    
    X_res, y_res = pipeline.fit_resample(xtrain, ytrain)
        #X_res, y_res =xtrain, ytrain
    print("Before sampling %s" % Counter(ytrain))
    print('Resampled dataset shape %s' % Counter(y_res))
        
    #model

    model=xgb.XGBRFClassifier(
        use_label_encoder=False, 
            scale_pos_weight=0.9,
            n_estimators=70,
            max_depth =6,
            n_jobs=-1,
            subsample=0.4,
            num_parallel_tree=20,
            eval_metric ='logloss',
            tree_method ='auto',
            objective='reg:logistic',
            gamma=.1,
            min_child_weight=6,
            booster='dart',
            eta =0.8
            )
    #fit the model on training data 
    model.fit(X_res,y_res)
    # make predictions
    preds = model.predict(xtest)
    preds_proba=model.predict_proba(xtest)[:,1]   
    # print('preds shape',preds_proba.shape) 
    
    t1=time.time()
    total_time = t1-t0    
    print('time to fit model:', total_time)
       
    accuracy_score = np.sum(preds == ytest) / len(ytest)       
         #log_loss= metrics.log_loss(train_df.OpenStatus,preds)
        
    #print(f"Fold:{fold_}")
    #print(f"Accuracy={accuracy_score}")
    conf_m=confusion_matrix(ytest,preds)
        #print('Confusion matrix\n',conf_m)
    roc_score=roc_auc_score(ytest, preds_proba)
    print('ROC AUC score\n', roc_score)
    t=[fold_,roc_score]
    total_conf.append(conf_m)
    total_roc.append(t)
    test_df.loc[:,"xgb_pred_n"] = preds_proba
    print('Confusion matrix\n', confusion_matrix(ytest,preds))

    return test_df[["id","target","kfold","xgb_pred_n"]], np.mean(total_roc,axis=0)[1] 



if __name__=="__main__":
    dfs =[]
    total_auc=[]
    for fold in range(5):
        temp_df, auc =run_training(fold)
        dfs.append(temp_df)
        total_auc.append(auc)
    fin_valid_df=pd.concat(dfs)
    print("******")
    print("preds shape",fin_valid_df.shape)
    print(total_auc)
    print("mean auc",np.mean(total_auc))
    fin_valid_df.to_csv("../model_preds/xgb_n.csv",index=False)
    print("******")
   
     