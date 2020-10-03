import time
from random import randint

from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.svm import SVC
import xgboost
from sklearn.inspection import permutation_importance

from helper.plots_and_scores import *


class RandomForestModel:
    """ Runs Random forest
        Parameter
        ---------
        X : input features
        y : target"""

    def __init__(self):
        self.random_forest = RandomForestClassifier(n_jobs=-1, criterion='gini')

    def defaultmodel(self, X, y):
        """
        Trains a model with default parameters and returns the model
        Returns
        --------
        un_opt_model: trained model

        """
        self.X = X
        self.y = y

        print("In unoptimized fit now--------")
        start_time = time.time()
        un_opt_model = self.random_forest.fit(self.X, self.y)
        time_to_run = time.time() - start_time
        print("Time taken to fit: {:.4f} s".format(time_to_run))
        return un_opt_model

    def RandomTune(self, X, y):
        """
        Trains a model using random parameters and returns the model

        Returns
        --------
        best_clf : trained model after optimization

        """
        # create random forest classifier model
        self.X = X
        self.y = y

        model = self.random_forest
        model_params = dict(
            # randomly sample numbers from 4 to 204 estimators
            n_estimators=[randint(120, 200)],
            # n_estimators =[80,100],
            # normally distributed max_features, with mean .25 stddev 0.1, bounded between 0 and 1
            # max_features=truncnorm(a=0, b=1, loc=0.25, scale=0.1),
            max_features=[0.3, 0.5, 0.8],
            # uniform distribution from 0.01 to 0.2 (0.01 + 0.199)
            # min_samples_split=uniform(0.01, 0.399),
            # uniform distribution from 0.5 to 0.99 (0.5 + 0.99)
            # max_samples=uniform(0.5,0.49),
            max_samples=[0.8, 0.9, 0.99],
            min_samples_leaf=[2, 3, 5]
        )
        # Set up the scoring method
        scorer = make_scorer(MccMetric, greater_is_better=True)
        # Set up the random search estimator, this will train 5 models.
        clf = RandomizedSearchCV(model, model_params, n_iter=2, random_state=0, scoring=scorer)
        print("In random tune fit now--------")
        start_time = time.time()
        fit_model = clf.fit(self.X, self.y)
        time_to_run = time.time() - start_time
        print("Time taken to fit: {:.4f} s".format(time_to_run))
        # Get the best estimator
        best_clf = fit_model.best_estimator_
        # Print the best parameters
        # pprint(fit_model.best_estimator_.get_params())
        return best_clf


class LogisticRegModel:
    """ Trains a Logistic Regression model"""

    def __init__(self):
        self.clf = LogisticRegression(max_iter=1000, C=100000, penalty='l2')

    def defaultmodel(self):
        # takes in X and y and trains a model with default parameters and returns the model
        self.X = X
        self.y = y
        print("In unoptimized fit now in LR--------")
        t = time.time()
        un_opt_model = self.clf.fit(self.X, self.y)
        elapsed_time = time.time() - t
        print("Time taken to fit: {:.4f} s".format(elapsed_time))
        print('\n')
        # un_opt_pred =un_opt_model.predict(self.X)
        return un_opt_model


class AdaBoostModel:
    def __init__(self):
        self.clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1),
                                      n_estimators=200, algorithm="SAMME.R",
                                      learning_rate=0.5)

    def defaultmodel(self, X, y):
        self.X = X
        self.y = y
        print("In unoptimized fit now in AdaBoost--------")
        t = time.time()
        model = self.clf.fit(self.X, self.y)
        elapsed_time = time.time() - t
        print("Time taken to fit: {:.4f} s".format(elapsed_time))
        print('\n')
        # un_opt_pred =un_opt_model.predict(self.X)
        return model


class XgboostModel:
    def __init__(self):
        self.clf = xgboost.XGBRFClassifier()

    def defaultmodel(self, X, y):
        self.X = X
        self.y = y
        print("In unoptimized fit now in XGBoost--------")
        t = time.time()
        model = self.clf.fit(self.X, self.y)
        elapsed_time = time.time() - t
        print("Time taken to fit: {:.4f} s".format(elapsed_time))
        print('\n')
        return model


class FeatImportance:
    def __init__(self, model, X, y):
        self.model = model
        self.X = X
        self.y = y

    def feat_importance(self):
        print("In feature importance now --------")
        t = time.time()
        result = permutation_importance(self.model, self.X, self.y, n_repeats=10, random_state=0)
        sorted_idx = result.importances_mean.argsort()
        elapsed_time = time.time() - t
        print("Time taken for feature importance: {:.4f} s".format(elapsed_time))
        return result, sorted_idx


class VotingCls:


    def __init__(self):
        self.log_clf = LogisticRegression(max_iter=1000, C=100000, penalty='l2')
        self.xgb_clf = xgboost.XGBRFClassifier()
        self.random_forest_clf = RandomForestClassifier(n_jobs=-1, criterion='gini',max_features=0.8,
                                                        n_estimators=140, min_samples_leaf=2)
        #self.svc_clf = SVC()
        #self.ADB_clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), n_estimators=200,
        #                                  algorithm="SAMME.R", learning_rate=0.5)

    def defaultmodel(self, X, y):
        print("In Voting classifier")
        self.X = X
        self.y = y
        voting_clf = VotingClassifier(estimators=[('lr', self.log_clf),('xbf',self.xgb_clf),
                                                    ('rf', self.random_forest_clf)],voting='soft')
        t = time.time()
        model = voting_clf.fit(self.X, self.y)
        elapsed_time = time.time() - t
        print("Time taken to fit voting classifier: {:.4f} s".format(elapsed_time))
        print('\n')
        return model


MODELS = {
    "randomforest": RandomForestModel(),
    "logisticregression": LogisticRegModel(),
    "adaboost": AdaBoostModel(),
    "xgboost": XgboostModel(),
    "votingclassifier": VotingCls()
}
