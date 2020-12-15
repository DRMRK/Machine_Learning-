# create folds.py
# import pandas and model_selection module from scikit-learn
import os
import pandas as pd
from sklearn import model_selection
CLEANED_DATA = os.environ.get("CLEANED_DATA")
KFOLD_DATA = os.environ.get("KFOLD_DATA")

if __name__=="__main__":
    # Read trainig data
    df = pd.read_feather(CLEANED_DATA)

    # create a new column called fold and fill it with -1
    df["kfold"] = -1

    # Randomize the rows of data

    df = df.sample(frac=1).reset_index(drop=True)

    # fetch labels
    y= df.NextLegBunchingFlag.values

    #initialize the kfold class from model_selection module

    kf = model_selection.StratifiedKFold(n_splits=5)

    # fill the new kfold column

    for f, (t_, v_) in enumerate(kf.split(X=df, y=y)):
        df.loc[v_, 'kfold'] = f
    # save the new csv with kfold column

    df.to_feather(KFOLD_DATA)    