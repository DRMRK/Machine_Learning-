

class RF_model:
    """ Runs Random forest """

    def __init__(self):
        self.random_forest = RandomForestClassifier(n_jobs=-1, criterion='gini')

    def unoptimized_RF(self, X, y):
        """
        Trains a model with default parameters and returns the model
        Parameter
        ---------
        X : input features
        y : target

        Returns
        --------
        un_opt_model: trained model

        """

    self.X = X
    self.y = y
    print("In unoptimized fit now--------")
    t = time.process_time()
    un_opt_model = self.random_forest.fit(self.X, self.y)
    elapsed_time = time.process_time() - t
    print("Time taken to fit: {:.4f} s".format(elapsed_time))
    print('\n')
    # un_opt_pred =un_opt_model.predict(self.X)
    return un_opt_model


def randomtuned_RF(self, X_train, y_train):
    """
    Trains a model using random parameters and returns the model
    Parameter
    ---------
    X : input features
    y : target

    Returns
    --------
    best_clf : trained model

    """


self.X_train = X_train
self.y_train = y_train
# create random forest classifier model
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
# Set up the random search estimator, this will train 5 models.
clf = RandomizedSearchCV(model, model_params, n_iter=5, random_state=0, scoring=scorer)
print("In randomtune fit now--------")
t = time.process_time()
fit_model = clf.fit(self.X_train, self.y_train)
elapsed_time = time.process_time() - t
print("Time taken to fit: {:.4f} s".format(elapsed_time))
print('\n')
# Get the best estimator
best_clf = fit_model.best_estimator_
# Print the best parameters
# pprint(fit_model.best_estimator_.get_params())
return best_clf