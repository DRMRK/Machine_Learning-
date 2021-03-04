#### Directory structure
.
|-----input

            |-----embeddings(put embeddings here)

            |-----raw_data(put raw data here)

            |-----tiny_data(put processed data here)

|-----src_fasttext

|-----LSTM_src (trained on aws)

|-----notebooks

|-----fastai_src (trained on google colab)

#### Order of operation
- cd src_fasttext
- sh run.sh, this runs the following in order: 

1/ a_select_tiny_data.py 
- reads whole datafile, selects the text columns and the target.
- does stratified sampling, saves fraction of data in "train_tiny.csv"

2/ b_create_embedding.py
- reads "train_tiny.csv"
- processes the text data (cleans, tokenizes)
- reads the fasttext embedding vectors
- converts tokens of the text field into 300 dimensional vectors.
- for each row in the dataframe averages the tokens, so for each row
we end up with a 300 dimensioal vector.
- since we have 2 columns we save two files with dimension (# of rows, 300)
- we also save the target column in a separate file.
- The net result of running this is three files --> BodyMarkDown.h5, title.h5, target.h5

3/ c_concatenate_vectors.py

- reads BodyMarkDown.h5, title.h5, target.h5
- concatenates above three  
- outputs full_data.h5

4/ d_create_kfold.py

- reads full_data.h5
- adds a k_fold column
- creates stratified kfold and saves it in full_data_folds.h5
- This dataset contain 'embedded text1','embedded text2','target','kfold'.
- dimension of the dataset is (# of rows, 602)
- this is the datset that we use for modelling 

5/ python lr_embedding.py

6/ python lr_embedding_for_blend.py (This saves the results in lr.csv with columns id,target,kfold,lr_pred )

7/ python lr_embedding_bagging_for_blend.py (This saves results in lr_bagging.csv with columns id,target,kfold,lr_bagging_pred)

8/ python GNB_embedding_blend.py (This saves results in GNB.csv with columns id,target,kfold,GNB_pred)

9/ python xgb_embedding_blend.py with different sampling strategy. save in xgb_n.csv and xgb_p.csv

10/ python blend_with_lr.py