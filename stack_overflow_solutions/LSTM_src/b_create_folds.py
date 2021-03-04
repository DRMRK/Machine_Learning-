import pandas as pd 
from sklearn import model_selection
import time

if __name__=="__main__":
    t0 = time.time()
    #read training data
   # df = pd.read_hdf(
    #    "../input/tiny_data/train_tiny.h5",
    #    key='dataset',
    #    mode ='r'
    #    )
    df=pd.read_csv("../LSTM_input/train_tiny.csv")
    print("data shape",df.shape)
    print('columns',df.columns)
    print("target\n",df.OpenStatus.value_counts())    
    # map alues to 1 and 0
    #df.OpenStatus = df.OpenStatus.apply(
    #    lambda x:1 if x=="open" else 0
    #    )
    # create a new column 
    df["kfold"] = -1
    # randomize the rows of data
    df = df.sample(frac=1).reset_index(drop=True)
    # get labels
    y = df.OpenStatus.values

    # initiate the kfold class 
    kf = model_selection.StratifiedKFold(n_splits=5)

    # fill kfold column

    for f, (t_,v_) in enumerate(kf.split(X=df,y=y)):
        df.loc[v_,'kfold'] = f 
    # save new h5
    #df.to_hdf(
    #    "../LSTM_input/train_tiny_folds_lstm.h5",
    #    key="dataset",
    #    mode ="w"
    #    ) 
    df.to_csv("../LSTM_input/train_tiny_folds_lstm.csv", index=False)    
    t1=time.time()
    print("Time taken", t1-t0)   