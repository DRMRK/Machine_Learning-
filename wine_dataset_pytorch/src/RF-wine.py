import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn import preprocessing


def run(fold):

    df = pd.read_csv("../input/train_folds.csv")
    #print("classes", df.wclass.value_counts())
    #df['quality']=(df['quality']-df['quality'].min()).astype('category')
    features =[
            f for f in df.columns if f not in ("quality", "kfold","wclass","wclass_num")
        ]
    # get trainign data using folds
    df_train=df[df.kfold !=fold].reset_index(drop=True)
    # get validation data using folds
    df_valid=df[df.kfold ==fold].reset_index(drop=True)

    scaler = preprocessing.StandardScaler()
    x_train = scaler.fit_transform(df_train[features])
    x_valid = scaler.transform(df_valid[features])
    #x_train = df_train[features]
    #x_valid = df_valid[features]

    model = RandomForestClassifier(n_estimators=2800, min_samples_leaf=1, n_jobs=-1,
    min_samples_split=2, class_weight="balanced_subsample",max_features=0.6, max_depth=40)
        
    # fit model on trainig data
    model.fit(x_train, df_train.wclass_num.values)

    #predict on validation data

    valid_preds = model.predict(x_valid)
        
    # get roc auc curve
    accuracy= metrics.accuracy_score(df_valid.wclass_num.values, valid_preds, normalize=False)
    balanced_accuracy = metrics.balanced_accuracy_score(df_valid.wclass_num.values, valid_preds)

    print(f"fold ={fold}, total correct ={accuracy}, balanced acc ={balanced_accuracy}, %correct ={accuracy/len(valid_preds)}")  
if __name__=="__main__":
    for fold in range(5):
        run(fold)    