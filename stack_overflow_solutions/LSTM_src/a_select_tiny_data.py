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
    
    # add a id column and set index values
    df ["id"] =df.PostId
    # take a small subset of data

    df ["question"] = df["Title"] + ' ' + df["BodyMarkdown"]
    df.OpenStatus = df.OpenStatus.apply(
        lambda x:1 if x=="open" else 0
        )
    # add a id column and set index values
    #df ["id"] =df.id
    # take a small subset of data

    #df ["question"] = df["Title"] + ' ' + df["BodyMarkdown"]

    Xdata = df[["id","question","OpenStatus"]]
    ydata = df[["OpenStatus"]]
    
    # choose part of the data
    X_train, X, y_train, y = train_test_split(
        Xdata, ydata, 
        test_size=0.01,
        random_state=42, 
        stratify=ydata
        )
    print("saving")

    # save the data, this contains the target column 
   # X.to_hdf(
    #    "../input/tiny_data/train_tiny.h5", 
     #   key='dataset',
     #   mode ='w',
     #   index= True
     #   )
    X.to_csv("../LSTM_input/train_tiny.csv",index=False)
    print("data shape",X.shape)
    print("done")