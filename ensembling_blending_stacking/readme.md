### Ensembling multiple models.
- I use the Quora comments dataset from Kaggle. Goal is to predict if two questions are same or different. I follow an approach outlined by Abhisek Thakur. 

#### Description from Kaggle:
The goal of this competition is to predict which of the provided pairs of questions contain two questions with the same meaning. The ground truth is the set of labels that have been supplied by human experts. The ground truth labels are inherently subjective, as the true meaning of sentences can never be known with certainty. Human labeling is also a 'noisy' process, and reasonable people will disagree. As a result, the ground truth labels on this dataset should be taken to be 'informed' but not 100% accurate, and may include incorrect labeling. We believe the labels, on the whole, to represent a reasonable consensus, but this may often not be true on a case by case basis for individual items in the dataset.

- There are 404290 rows of data (with columns id, qid1, qid2, question1, question2, is_duplicate)
- To numericalize the text data I use TFIDF vectorizer and Countvectorizer.  
- Data obtainend after using the two numericalization methods are separately fed to the classification models.  
- The Goal is to predict the "is_duplicate" column.
- I use 5 fold cross validation to judge the models based on the average AUC score.

#### Results 
##### Using TFIDF
- Logistic regression --> AUC score 0.8197
- Gaussian Naive Bayes --> AUC score 0.6542
##### Using TFIDF and Truncated Singular value decomposition.
- I use truncated SVD to make the resultant sparse matrix from TFIDF suitable for Random Forest. 
- Random Forest --> AUC score 0.8294
##### Using CountVectorizer 
- Logistic regression --> AUC score 0.7676   

##### Stacking using XGboost 
- I use the output of the above 4 models as features and train a  XGboost classfier model on these features. 
- AUC score 0.8563 

##### From this exercise we see that stacking helps in improving the AUC score.  