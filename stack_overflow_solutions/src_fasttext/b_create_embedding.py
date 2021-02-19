"""
Here we use the fasttext embeddings and 
convert our sentences into 300 dimensional vectors
"""
import numpy as np
import pandas as pd
from numpy import save
from numpy import load

from sklearn import model_selection

import nltk
from nltk.tokenize import word_tokenize
#nltk.download('stopwords')

import re, string
import io
import sys

import time

#punctuation = list(string.punctuation)

my_stopwords = nltk.corpus.stopwords.words('english')# punctuation

re_tok = re.compile(f'([{string.punctuation}“”¨«»®´·º½¾¿¡§£₤‘’])')

def remove_links(tweet):
    '''Takes a string and removes web links from it'''
    tweet =str(tweet).lower()
    tweet = re.sub(r'http\S+', '', tweet) # remove http links
    tweet = re.sub(r'bit.ly/\S+', '', tweet) # rempve bitly links
    tweet = tweet.strip('[link]') # remove [links]
    return tweet

def tokenize(s):
    s=str(s).lower()
    output = re.sub(r'\d+', '', s) # remove numbers 
    output = remove_links(output)
    result = re_tok.sub(r' \1 ', output).split() 
    result = [word for word in result if len(word)>2]
    result = [word for word in result if word not in my_stopwords]
    return result

# convert sentences to vectors from embedding, embedding is 300 dimensional
def sentence_to_vec(s,embedding_dict,stop_words,tokenizer):
    """
    s: sentence, string
    embedding_dict: dictionary word: vector
    stop_words: list of stop words
    tokenizer: tokenizer function
    """
    # tokenize the sentence
    words = s
    words = tokenizer(words)
    
    # keep only alpha numeric tokens
    words =[w for w in words if w.isalpha()]
    # initialize empty list to store embeddings
    M = []
    for w in words:
        # for every word, get the embedding from the dictionary
        # and append to the list of embeddings
        if w in embedding_dict:
            M.append(embedding_dict[w])
    # if we don't have any vectors return zeros
    if len(M)==0:
        return np.zeros(300)
    # convert list of embeddings to array
    M = np.array(M)
    # calculate sum over axis=0
    v = M.sum(axis=0)
    return v/np.sqrt((v**2).sum())       


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


#create sentence embeddings
def get_vectors(vectors,column,embeddings):
    """
    vectors: empty array to fill in
    column: dataframe column
    return embedding vectors
    """
    for body in column:
        vectors.append(
            sentence_to_vec(s= body,
                           embedding_dict = embeddings,
                           stop_words =my_stopwords,
                           tokenizer=tokenize
                           )
        )
    return vectors  


def get_sentence_embedding(df,column=None):    
    """
    param
    df: dataframe
    column: datafrane column to embedd
    return: dataframe with embedding.dim (#of rows, 301)
    """
    #Create empty dataframe
    
    data=pd.DataFrame([])      
        
    # loop over the postid
    
    for i in df.PostId:
        B=df.loc[df.PostId==i]    
        B=B[column].values
        # get the embeddings
        X= sentence_to_vec(s=[B],
                           embedding_dict = embeddings,
                           stop_words =my_stopwords,
                           tokenizer=tokenize
                           )
        
        data1=pd.DataFrame(data=X)
        data1=data1.T
        data1=data1.add_suffix('_'+column)
        #add id column
        data1["id"]=i
        data1.columns=data1.columns.astype(str)
        data=data.append(data1)        
    return data

def get_target(df,column=None):
    
    data=pd.DataFrame([])
    
    for i in df.PostId:
        B=df.loc[df.PostId==i]    
        B=B[column].values
        data1=pd.DataFrame(data=B)
        data1=data1.T
        #add id column
        data1["id"]=i
        data1.columns=data1.columns.astype(str)
        data1=data1.rename(columns={'0':'target'})
        data=data.append(data1)        
    return data 

if __name__=="__main__":

    t0_total =time.time()

    # Read embeddings
    t0=time.time()
    print("Loading embeddings")
    embeddings = load_vectors("../input/embeddings/crawl-300d-2M.vec")  
    
    t1 =time.time()
    total_time=t1-t0
    print("time to load", total_time)
    print(" ")
    
    t0=time.time()

    # Read trainig data
    
    df= pd.read_hdf('../input/tiny_data/train_tiny.h5',key='dataset')
    df = df[["PostId","Title","BodyMarkdown","OpenStatus"]]
    y = df.OpenStatus.values

    t1 =time.time()
    total_time=t1-t0
    print("time to read", total_time)
    print(" ")

    # create a new column called fold and fill it with -1

    t0=time.time()
    print("first column embedding")

    # First column "BodyMarkdown"
    BodyMarkDown_appended = get_sentence_embedding(df,'BodyMarkdown')

    print(BodyMarkDown_appended.shape)
    #save embeddings
    BodyMarkDown_appended.to_hdf(
        "../input/tiny_data/BodyMarkDown.h5",
        key='dataset',
        mode ='w',
        index= False
        )
    t1=time.time()
    print("time to embed",t1-t0)    

    # second column "Title"
    print("second column embedding")
    t0=time.time()
    title_appended = get_sentence_embedding(df,'Title') 
    print(title_appended.shape)
    title_appended.to_hdf(
        "../input/tiny_data/title.h5",
        key='dataset',
        mode ='w',
        index= False
        )
    t1=time.time()
    print("time to embed", t1-t0)
    print(" ")

    # target
    target = get_target(df,"OpenStatus")
    print(target.shape)
    target.to_hdf(
    "../input/tiny_data/target.h5",
    key='dataset',
    mode ='w',
    index= False)

    t1_total =time.time()
    print(f"total time for the process {df.shape},{t1_total-t0_total}")
    print("---------")