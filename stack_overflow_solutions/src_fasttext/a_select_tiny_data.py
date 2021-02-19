"""
Select tiny part of data

    # reads the whole datafile
    # creates a tiny subset of data with 
    same traget distribution as the original data

"""
import pandas as pd
from sklearn.model_selection import train_test_split

if __name__=="__main__":
    
    df = pd.read_csv('../input/raw_data/train.csv')
    # take a small subset of data

    Xdata = df[["PostId","Title", "BodyMarkdown","OpenStatus"]]
    ydata = df[["OpenStatus"]]
    
    # choose part of the data
    X_train, X, y_train, y = train_test_split(
        Xdata, ydata, 
        test_size=0.01,
        random_state=42, 
        stratify=ydata
        )
    replace = X.loc[:,"OpenStatus"].apply(
        lambda x: 'open' if x=='open' else 'closed'
        )
    X.loc[:,"OpenStatus"]= replace 
    print("saving")

    # save the data, this contains the target column 
    X.to_hdf(
        "../input/tiny_data/train_tiny.h5", 
        key='dataset',
        mode ='w',
        index= False
        ) 
    print("data shape",X.shape)
    print("done")