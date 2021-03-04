import io
import torch 

import numpy as np 
import pandas as pd 

import tensorflow as tf 

from sklearn import metrics 
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_auc_score

import config 
import dataset 
import engine 
import lstm 


def load_vectors(fname):
    fin = io.open(
        fname,'r',encoding ='utf-8',
        newline = '\n',
        errors='ignore'
        )
    n,d = map(int,fin.readline().split())
    data ={}
    for line in fin:
        tokens = line.rstrip().split(' ')
        data[tokens[0]]=list(map(float,tokens[1:]))
    return data 

def create_embedding_matrix(word_index, embedding_dict):
    """
    Creates embedding matrix
    :param word_index: dictionary with word:index_value
    :paran embedding_dict: a dictionary with word:embedding_vector
    :return: numpy array with embedding vectors for all known words 
    
    """
    #initialize matrix with zeros
    embedding_matrix = np.zeros((len(word_index)+1,300))
    # loop over all the words 
    for word, i in word_index.items():

        # if word is found in pre-trained embedding
        # update the matrix, if not vector is zero     
        if word in embedding_dict:
            embedding_matrix[i]= embedding_dict[word]
        return embedding_matrix

def run(df, fold):
    """
    Run training and validation for a given fold
    :param df: dataframe with kold column
    :param fold: current fold, int
    """        
    # training dataframe
    train_df = df[df.kfold!=fold].reset_index(drop=True)
    # validation dataframe 
    valid_df = df[df.kfold==fold].reset_index(drop=True)

    print("Fitting tokenizer")
    # tokenize
    tokenizer = tf.keras.preprocessing.text.Tokenizer()
    tokenizer.fit_on_texts(df.BodyMarkdown.values.tolist())

    # convert training data to sequence 
    xtrain = tokenizer.texts_to_sequences(train_df.BodyMarkdown.values)
    # convert validation data to sequence 
    xtest = tokenizer.texts_to_sequences(valid_df.BodyMarkdown.values)
    # zero pad the trainign sequence, padding on left side
    xtrain = tf.keras.preprocessing.sequence.pad_sequences(
        xtrain, maxlen=config.MAX_LEN
        )
    # zero pad validation sequence
    xtest = tf.keras.preprocessing.sequence.pad_sequences(
        xtest, maxlen=config.MAX_LEN
        )    
    # initialize dataset class for training 
    train_dataset = dataset.QUORADataset(
        BodyMarkdown=xtrain,
        OpenStatus=train_df.OpenStatus.values
        )    

    # create torch DataLoader
    train_data_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size = config.TRAIN_BATCH_SIZE,
        num_workers =2
        )

    #initialize dataset class for validation 
    valid_dataset = dataset.QUORADataset(
        BodyMarkdown=xtest,
        OpenStatus=valid_df.OpenStatus.values
        ) 
     # create torch DataLoader
    valid_data_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size = config.VALID_BATCH_SIZE,
        num_workers =1
        )           

    print("loading embeddings")
    embedding_dict = load_vectors("../input/embeddings/crawl-300d-2M.vec")
    embedding_matrix = create_embedding_matrix(
        tokenizer.word_index,embedding_dict
        )
    # create torch device

    device = torch.device("cpu")
    # get LSTM model
    model = lstm.LSTM(embedding_matrix)
    # send model to device
    model.to(device) 
    #initialize adam optimizer 
    optimizer = torch.optim.Adam(model.parameters(), lr = 1e-3)
    print("Training model")
    # best accuracy to 0
    best_accuracy = 0
    # early stopping counter 
    early_stopping_counter = 0
    # train and validate for all epoch 
    for epoch in range(config.EPOCHS):
        # train one epoch 
        engine.train(train_data_loader, model, optimizer, device)
        # validate
        outputs, targets = engine.evaluate(
            valid_data_loader, model, device
            )
        # threshold 
        outputs1 = np.array(outputs)>= 0.5
        # calculate accuracy 
        accuracy = metrics.accuracy_score(targets,outputs1)
        conf_m=confusion_matrix(targets,outputs1)

        print(
            f"Fold:{fold}, Epoch:{epoch}, Accuracy_score ={accuracy}"
            )
        print("conf_m\n",conf_m)
        roc_score=roc_auc_score(targets, np.array(outputs))
        print('ROC AUC score\n', roc_score)
        print("---")
        # early stopping 
        if accuracy > best_accuracy:
            best_accuracy=accuracy
        else:
            early_stopping_counter+=1
        
        if early_stopping_counter >2:
            break

if __name__ == "__main__":
    #load data 
    df = pd.read_hdf(
        path_or_buf="../LSTM_input/full_data_folds_lstm.h5",
        key='dataset'
        )                               

    # train for all folds 
    run(df,fold=0)    