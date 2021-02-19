import numpy as np
import pandas as pd

from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt

import nltk
from nltk.tokenize import word_tokenize
#nltk.download('stopwords')

from sklearn import linear_model
from sklearn import metrics
from sklearn import model_selection
from sklearn.preprocessing import StandardScaler

import re, string
import io

import time




if __name__=="__main__":

    total_roc = []
    total_conf =[]
    
    t0=time.time()
    #df = pd.read_csv("../input/embedded_train_tiny_folds.csv")
    df = pd.read_hdf(
        path_or_buf="../input/tiny_data/full_data_folds.h5",
        key='dataset'
        )
    print("tg\n",df.target.value_counts())   
    print(" ") 
    print('shape',df.shape)
    t1=time.time()
    total_time = t1-t0
   # print("time to read file",total_time)

    for fold_ in range(5):
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
        
        sc = StandardScaler()
        sc.fit(xtrain)

        xtrain = sc.transform(xtrain)
        xtest = sc.transform(xtest)
        
        # target
        # First make the target binary
        train_df.target = train_df.target.apply(
            lambda x:'open' if x=='open' else 'closed'
            )

        test_df.target = test_df.target.apply(
            lambda x:'open' if x=='open' else 'closed'
            )    
        ytrain = train_df.target
        ytest = test_df.target
        
        #model

        model=linear_model.LogisticRegression(
            penalty ='l2',
            C=.8e-4,
            max_iter=5000,
            class_weight='balanced',
            solver='liblinear'
            )
        #fit the model on training data 
        model.fit(xtrain,ytrain)
        # make predictions
        preds = model.predict(xtest)
        preds_proba=model.predict_proba(xtest)[:,1]   
      #  print('preds shape',preds_proba.shape) 
    
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
        #print(" ")
        
    for i in range(5):
        #print("ROC_AUC\n",total_roc[i])
        print("Confusion_matrix\n",total_conf[i])
        #print("mean_score")
        print("-----")    

    print(np.mean(total_roc,axis=0)[1])       