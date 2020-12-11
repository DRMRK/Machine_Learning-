import pandas as pd

from sklearn import linear_model
from sklearn import metrics
from sklearn import preprocessing

def run(fold):

    df = pd.read_csv("../input/train_folds.csv")
    features =[
            f for f in df.columns if f not in ("quality", "kfold",'wclass',"wclass_num")
        ]
    # get trainign data using folds
    df_train=df[df.kfold !=fold].reset_index(drop=True)
    # get validation data using folds
    df_valid=df[df.kfold ==fold].reset_index(drop=True)

    scaler = preprocessing.StandardScaler()
    x_train = scaler.fit_transform(df_train[features])
    # Transform using train fit scaler 
    x_valid = scaler.transform(df_valid[features])

    model = linear_model.LogisticRegression(max_iter=500, C=1000)
        
    # fit model on trainig data
    model.fit(x_train, df_train.wclass_num.values)

    #predict on validation data

    valid_preds = model.predict(x_valid)
        
    # get roc auc curve
    accuracy= metrics.accuracy_score(df_valid.wclass_num.values, valid_preds,normalize=False)
    #balanced_accuracy = metrics.balanced_accuracy_score(df_valid.wclass_num.values, valid_preds)
    print(f"fold ={fold}, total correct ={accuracy}, #of valid data points ={len(valid_preds)} %correct ={accuracy/len(valid_preds)}")    
if __name__=="__main__":
    for fold in range(5):
        run(fold)    