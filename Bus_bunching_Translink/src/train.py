# We use this to analyze the data cleaned in clean_data.py

from helper.structured import *
from sklearn.model_selection import train_test_split
from helper.plots_and_scores import *
import time
from sklearn.preprocessing import StandardScaler
from . import dispatcher

TRAINING_DATA = os.environ.get("TRAINING_DATA")
CLEANED_DATA = os.environ.get("CLEANED_DATA")
report1 = os.environ.get('REPORT1')
MODEL = os.environ.get("MODEL")

if __name__ == '__main__':
    # Load the cleaned dataset
    data = pd.read_feather(CLEANED_DATA)
    # There is an extra index column, remove it
    data = data.drop(['index'], axis=1)
    # Convert all columns to numeric column and
    # separate out the target variable

    X, y, nas = proc_df(data, y_fld='NextLegBunchingFlag',
                        skip_flds=['NextNextLegBunchingFlag',
                                   'NextNextNextLegBunchingFlag'])
    # Stratified sampling for train, test and validation datasets
    X = StandardScaler().fit_transform(X)
    X_train, X_test_valid, y_train, y_test_valid = \
        train_test_split(X, y, random_state=1, test_size=0.20,
                         stratify=y)
    X_test, X_valid, y_test, y_valid = \
        train_test_split(X_test_valid, y_test_valid,
                         random_state=1, test_size=0.50,
                         stratify=y_test_valid)
    # Now the data is ready for modelling
    print("Instantiating and training the model")
    start_time = time.time()
    # choose the model
    model = dispatcher.MODELS[MODEL]
    # This steps returns the fitted model ready for prediction
    clf = model.defaultmodel(X_train, y_train)
    end_time = time.time() - start_time
    print("Time taken for training: {:.4f} s".format(end_time))
    print("Scores for test data")
    y_pred_test = clf.predict(X_test)
    scores_test = PlotsAndScores(y_test, y_pred_test, None)
    scores_test.print_scores()
    print('-------------\n')
    # scores_test.display_confusion_matrix(report1, 'Test')
    print("Scores for train data")
    y_pred_train = clf.predict(X_train)
    scores_train = PlotsAndScores(y_train, y_pred_train, None)
    scores_train.print_scores()