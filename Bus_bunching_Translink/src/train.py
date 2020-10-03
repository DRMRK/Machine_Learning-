# We use this to analyze the data cleaned in clean_data.py

from helper.structured import *
from sklearn.model_selection import train_test_split
from helper.plots_and_scores import *
from sklearn.preprocessing import StandardScaler

from src.dispatcher import FeatImportance
from . import dispatcher

TRAINING_DATA = os.environ.get("TRAINING_DATA")
CLEANED_DATA = os.environ.get("CLEANED_DATA")
report1 = os.environ.get('REPORT1')
MODEL = os.environ.get("MODEL")

if __name__ == '__main__':
    # Load the cleaned dataset
    data = pd.read_feather(CLEANED_DATA)
    # There is an extra index column, remove it
    data = data.drop(['index', 'NextNextLegBunchingFlag',
                      'NextNextNextLegBunchingFlag'], axis=1)
    # Convert all columns to numeric column and
    # separate out the target variable

    X, y, nas = proc_df(data, y_fld='NextLegBunchingFlag')
    # Standardize the dataset
    ss = StandardScaler()
    X_scaled = pd.DataFrame(ss.fit_transform(X), columns=X.columns)
    # Stratified sampling for train, test and validation datasets
    X_train, X_test_valid, y_train, y_test_valid = \
        train_test_split(X_scaled, y, random_state=1, test_size=0.20,
                         stratify=y)
    X_test, X_valid, y_test, y_valid = \
        train_test_split(X_test_valid, y_test_valid,
                         random_state=1, test_size=0.50,
                         stratify=y_test_valid)

    # Now the data is ready for modelling
    print("Instantiating and training the model")
    # choose the model
    model = dispatcher.MODELS[MODEL]
    # This steps returns the fitted model ready for prediction
    clf = model.RandomTune(X_train, y_train)
    print('-------------')
    print("Scores for train data")
    y_pred_train = clf.predict(X_train)
    scores_train = PlotsAndScores(y_train, y_pred_train, None)
    scores_train.print_scores()
    print('-------------')
    print("Scores for test data")
    y_pred_test = clf.predict(X_test)
    scores_test = PlotsAndScores(y_test, y_pred_test, None)
    scores_test.print_scores()
    # scores_test.display_confusion_matrix(report1, 'Test')
    print('-------------')
    print("Now do feature selection")
    imp_feature = FeatImportance(clf, X_test, y_test)
    result, sorted_idx = imp_feature.feat_importance()
    print('Top 20 features later ones are more important: ', X_test.columns[sorted_idx[-20:]])
    scores_test.feature_importance_plot(result, data, sorted_idx)

    # Choose the top 8 features and run the predictions again
    feat = X_test.columns[sorted_idx[-20:]]
    top_8_features = [i for i in feat]
    X_scaled_chosen = X_scaled[top_8_features]
    print('-------------')
    print('Now we choose top 20 features and run the model again')
    # Stratified sampling for train, test and validation datasets
    X_train, X_test_valid, y_train, y_test_valid = \
        train_test_split(X_scaled_chosen, y, random_state=1, test_size=0.20,
                         stratify=y)
    X_test, X_valid, y_test, y_valid = \
        train_test_split(X_test_valid, y_test_valid,
                         random_state=1, test_size=0.50,
                         stratify=y_test_valid)
    # This steps returns the fitted model ready for prediction
    clf = model.defaultmodel(X_train, y_train)
    # scores_test.display_confusion_matrix(report1, 'Test')
    print("Scores for train data")
    y_pred_train = clf.predict(X_train)
    scores_train = PlotsAndScores(y_train, y_pred_train, None)
    scores_train.print_scores()
    print('-------------')
    print("Scores for test data")
    y_pred_test = clf.predict(X_test)
    scores_test = PlotsAndScores(y_test, y_pred_test, None)
    scores_test.print_scores()
