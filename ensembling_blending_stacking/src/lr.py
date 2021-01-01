import pandas as pd
from sklearn import linear_model
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import metrics 

def run_training(fold):
    df = pd.read_csv("../data/train_folds.csv")
    df.text = df.text.apply(str)
    df_train = df[df.kfold!=fold].reset_index(drop=True)
    df_valid = df[df.kfold==fold].reset_index(drop=True)

    tvf = TfidfVectorizer()
    tvf.fit(df_train.text.values)
    
    xtrain=tvf.transform(df_train.text)
    xvalid =tvf.transform(df_valid.text)

    ytrain = df_train.is_duplicate
    yvalid = df_valid.is_duplicate

    clf = linear_model.LogisticRegression(random_state=0,max_iter=1500,C=1,solver='saga',n_jobs=-1,penalty='l2')
    clf.fit(xtrain,ytrain)
    pred = clf.predict_proba(xvalid)[:,1]

    auc = metrics.roc_auc_score(yvalid,pred)
    
    print(f"kfold = {fold}, auc ={auc}")

    df_valid.loc[:,"lr_pred"] = pred
    
    return df_valid[["id","is_duplicate","kfold","lr_pred"]]


if __name__=="__main__":
    dfs =[]
    for j in range(5):
        temp_df = run_training(j)
        dfs.append(temp_df)
    fin_valid_df = pd.concat(dfs)
    print(fin_valid_df.shape)   
    fin_valid_df.to_csv("../model_preds/lr.csv",index=False) 





