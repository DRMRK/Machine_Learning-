Here I keep track of small projects that I am doing to play with data, learn/pracice algorithms/topics  

## Airline_data_analysis
In  this data visualization project I download flight data from Bureau of Transport Statistics.
- I use pandas for data exploration.
- After cleaning the data I use plotly to make an interactive visualization.
- The result is a plot that shows the aiport locations on a MAP of US, by hovering the mouse on the plot one can get information such as total number of flights, fraction of delayed flights etc.   


## NLP_LDA 

In this Natural language processing project I use twitter data and try to identify prevalent topics in the tweets directed at Translink. I use two versions of Latent Dirichlet Allocation- Gensim implementation and Mallet implementation. Due to small sentences in twitter data this technique is not as effective as with longer documents but still gives some idea of the underlying topics. 

## classify_cars

In this computer vision project I use the Stanford cars dataset to identify cars usign their images. Details of this image classification problem are in https://ai.stanford.edu/~jkrause/cars/car_dataset.html

- I use Convolutional Neural Networks.
- I use I use transfer learning, where I use pretrained weights from RESNET50. 
- This was too slow in my computer so I trained the model on Google Cloud Platform. (Machine type 4 vCPUs, 15 GB RAM, NVIDIA Tesla T4X1)

## quora-comments 
This is a problem from Kaggle. The goal of the competition is to predict which of the provided pairs of questions contain two questions with the same meaning. The ground truth is the set of labels that have been supplied by human experts. There are 404290 rows of data (with columns id, qid1, qid2, question1, question2, is_duplicate)

#### To test my  models I use the provided labelled data. Goal is to predict if question1 and question2 are same.
- I divide the labelled data into training(70% of total) and test(30% of total).
- I judge the models on accuracy on test data.
- I use a bag of word model.
- I combine two questions into one column.
- I tokenize the combined column.
- I numericalize the combined column using TFIDF. The result of TFIDF is our input to the models.
- When I use logistic regresion for this classification problem I obtain score 0.796 in about 60 s.
- When I Standardize the features I see that training slows down and did not converge till 2500 iterations in about 25 minute.
- When I use NaiveBayes Logistic regression we obtain accurace score 0.796 in about 9 minute.
- When I use Xgboost we obtain accurace score 0.793 in about 9 minute.
- I tried Gaussian Naive Bayes and Support vector machine. Using a small subset of data I find traing is slow so did not move forward with these

## Cassava_Leaf_Disease_Classification

- Data from Kaggle
- Identify the type of disease present on a Cassava Leaf image. 
