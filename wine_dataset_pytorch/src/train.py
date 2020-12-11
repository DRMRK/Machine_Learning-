# env image-processing
import pandas as pd
from sklearn import metrics
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torch.nn.functional as F
import math
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing

import dataset
import config
import network
import engine

def run(df, fold):
    # select features
    features =[f for f in df.columns if f not in ("quality", "kfold","wclass","wclass_num")]
    # Normalize inputs 
    scaler = preprocessing.StandardScaler()
    # get training data using folds
    train_df=df[df.kfold !=fold].reset_index(drop=True)
    xtrain = scaler.fit_transform(train_df[features])
    ytrain = (train_df["wclass_num"]-train_df["wclass_num"].min()).astype('category')
    # get validation data using folds
    valid_df=df[df.kfold ==fold].reset_index(drop=True)
    xvalid = scaler.fit_transform(valid_df[features])
    yvalid = (valid_df["wclass_num"]-valid_df["wclass_num"].min()).astype('category')
    
    # initialize dataset class for training
   
    train_dataset = dataset.WineDataset(features=xtrain, target=ytrain)
    # Make troch dataloader for training 
    # torch dataloader loads the data using dataset class in batches 
    # specifed by bath_size
    train_data_loader = torch.utils.data.DataLoader(dataset = train_dataset,
    batch_size=config.TRAIN_BATCH_SIZE, num_workers=0)

    # initialize dataset class for vailidation
   
    valid_dataset = dataset.WineDataset(features=xvalid, target=yvalid)
    # Make troch dataloader for training 
    # torch dataloader loads the data using dataset class in batches 
    # specifed by bath_size
    valid_data_loader = torch.utils.data.DataLoader(dataset = valid_dataset,
    batch_size=config.VALID_BATCH_SIZE, num_workers=0)
    # create toech device
    device = torch.device("cpu")
    
    # get the model
    model = network.Network()
    # initialize the optimizer
    learning_rate = 0.00008

    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    for epoch in range(config.EPOCHS):
        total_correct = 0
        # train one epoch
        loss, train_correct  = engine.train(train_data_loader,model,optimizer, device)
        # calculate total correct on validation data 
        total_correct = engine.evaluate(valid_data_loader, model, device) 
        if epoch >(config.EPOCHS-5):
            print("epoch",epoch)
            print("loss: %.2f" %loss,    
                "valid_correct:", total_correct,
                "valid_correct_pcntg: %.4f" %(total_correct/xvalid.shape[0]),
                "train_correct_pcntg: %.4f" %(train_correct/xtrain.shape[0]),
                "train_correct", train_correct
                )
if __name__=="__main__":
    df=pd.read_csv('../input/train_folds.csv')
    for fold in range(5):
        print(f"fold {fold}")
        run(df, fold)
        print('----')




