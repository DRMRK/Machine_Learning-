## Predict if two quora questions are same
This is a Kaggle problem, to test our models we use labelled data.
- I divide the labelled data into training and test.
- I judge the models on accuracy on test data. #### Summary
- I first use a bow model.
- I combine two questions into one column.
- I tokenize the combined column.
- I numericalize the combined column using TFIDF. The result of TFIDF is our input to the models.
- When I use logistic regresion for this classification problem I obtain score 0.796 in about 60 s.
- When I Standardize the features I see that training slows down and did not converge till 2500 iterations in about 25 minute.
- When I use NaiveBayes Logistic regression we obtain accurace score 0.796 in about 9 minute.
- When I use Xgboost we obtain accurace score 0.793 in about 9 minute.
- I tried Gaussian Naive Bayes and Support vector machine. Using a small subset of data I find traing is slow so did not move forward with these models.
