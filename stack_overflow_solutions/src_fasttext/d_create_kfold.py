import pandas as pd
import numpy as np
import pandas as pd
from numpy import save
from numpy import load
import time
from sklearn import model_selection
from pathlib import Path


if __name__=="__main__":
    
    t0=time.time()
    
    df = pd.read_hdf("../input/tiny_data/full_data.h5",key='dataset',mode ='r')
    y = df.target.values
    print(df.target.value_counts())
    
    
    df.loc[:,"kfold"] =-1

    print("data loaded")
    skf=model_selection.StratifiedKFold(n_splits=5)
    
    for f,(t_,v_) in enumerate(skf.split(X=df,y=y)):
        df.loc[v_,"kfold"]=f
    
    #save data
    df.to_hdf("../input/tiny_data/full_data_folds.h5",key='dataset',mode ='w')    
    t1=time.time()
    print("Time taken",t1-t0)

    