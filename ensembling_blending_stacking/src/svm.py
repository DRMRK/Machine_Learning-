import pandas as pd
from sklearn import svm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import metrics 
from sklearn import decomposition
import re, string
re_tok = re.compile(f'([{string.punctuation}“”¨«»®´·º½¾¿¡§£₤‘’])')
def tokenize(s): return re_tok.sub(r' \1 ', s).split()

def run_training(fold):
    df = pd.read_csv("../data/train_folds.csv")
    df.text = df.text.apply(str)
    df_train = df[df.kfold!=fold].reset_index(drop=True)
    df_valid = df[df.kfold==fold].reset_index(drop=True)

    tvf = TfidfVectorizer(
        ngram_range=(1,2), tokenizer=tokenize,
        min_df=3, max_df=0.9, strip_accents='unicode', 
        use_idf=1,smooth_idf=1, sublinear_tf=1 
        )
    tvf.fit(df_train.text.values)
    
    xtrain=tvf.transform(df_train.text)
    xvalid =tvf.transform(df_valid.text)

    svd = decomposition.TruncatedSVD(n_components =100)
    svd.fit(xtrain)
    xtrain_svd =svd.transform(xtrain)
    xvalid_svd =svd.transform(xvalid)

    ytrain = df_train.is_duplicate
    yvalid = df_valid.is_duplicate

    clf = svm.LinearSVC()
    clf.fit(xtrain_svd,ytrain)
    pred = clf.predict(xvalid_svd)

    auc = metrics.roc_auc_score(yvalid,pred)
    
    print(f"kfold = {fold}, auc ={auc}")

    df_valid.loc[:,"svm_pred"] = pred
    
    return df_valid[["id","is_duplicate","kfold","svm_pred"]]


if __name__=="__main__":
    dfs =[]
    for j in range(5):
        temp_df = run_training(j)
        dfs.append(temp_df)
    fin_valid_df = pd.concat(dfs)
    print(fin_valid_df.shape)   
    fin_valid_df.to_csv("../model_preds/svm.csv",index=False) 