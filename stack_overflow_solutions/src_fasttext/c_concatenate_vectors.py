import pandas as pd
import numpy as np
import pandas as pd
from numpy import save
from numpy import load
import time

if __name__=="__main__":

    t0=time.time()
    
    #Load the embeddng vectors
    text1=pd.read_hdf("../input/tiny_data/title.h5",key='dataset',mode ='r')
    text2=pd.read_hdf("../input/tiny_data/BodyMarkDown.h5",key='dataset',mode ='r')
    target =pd.read_hdf("../input/tiny_data/target.h5",key='dataset',mode ='r')
    
    #Merge with id 
    full_data=text1.merge(text2,how='left',left_on='id',right_on='id')
    print(full_data.shape)
    full_data=full_data.merge(target,how='left',left_on='id',right_on='id')
    print(full_data.shape)
    
    #Save the data
    full_data.to_hdf(
        "../input/tiny_data/full_data.h5",
        key='dataset',
        mode ='w',
        index= False)
    t1=time.time()
    print("total time", t1-t0)