# We use this to analyze the data cleaned in clean_data.py

from helper.structured import *
from sklearn.model_selection import train_test_split
from helper.plots_and_scores import *
from sklearn.preprocessing import StandardScaler

from src.dispatcher import FeatImportance
from . import dispatcher

TRAINING_DATA = os.environ.get("TRAINING_DATA")
CLEANED_DATA = os.environ.get("CLEANED_DATA")
KFOLD_DATA = os.environ.get("KFOLD_DATA")
report1 = os.environ.get('REPORT1')
MODEL = os.environ.get("MODEL")

if __name__ == '__main__':
    # Load the cleaned dataset
    data = pd.read_feather(KFOLD_DATA)
    # There is an extra index column, remove it
    data = data.drop(['index', 'NextNextLegBunchingFlag',
                      'NextNextNextLegBunchingFlag'], axis=1)
    
    # Standardize the dataset
    scaler = StandardScaler()

    for fold in range(5):
        print(" " )
        print(f"fold {fold}")
         # get training data using folds
        train_df=data[data.kfold !=fold].reset_index(drop=True)
        #X_scaled = pd.DataFrame(ss.fit_transform(X), columns=X.columns)
        X_train, y_train, nas = proc_df(train_df, y_fld='NextLegBunchingFlag')
        X_train = pd.DataFrame(scaler.fit_transform(X_train),columns=X_train.columns)
        print("Train Total rows, Total features",X_train.shape)
        # get validation data using folds
        valid_df=data[data.kfold ==fold].reset_index(drop=True)
        X_valid, y_valid, nas = proc_df(valid_df, y_fld='NextLegBunchingFlag')
        X_valid = pd.DataFrame(scaler.transform(X_valid),columns=X_valid.columns)
        print("Valid Total rows, Total features",X_valid.shape)


    # Now the data is ready for modelling
        print("Instantiating and training the model")
    # choose the model
        model = dispatcher.MODELS[MODEL]
    # This steps returns the fitted model ready for prediction
        clf = model.defaultmodel(X_train, y_train)
        print('-------------')
        print("Scores for train data")
        y_pred_train = clf.predict(X_train)
        scores_train = PlotsAndScores(y_train, y_pred_train, None)
        scores_train.print_scores()
        print('-------------')
        print("Scores for test data")
        y_pred_valid = clf.predict(X_valid)
        scores_valid = PlotsAndScores(y_valid, y_pred_valid, None)
        scores_valid.print_scores()
        # scores_test.display_confusion_matrix(report1, 'Test')
        print('-------------')
        print("Now do feature selection")
        imp_feature = FeatImportance(clf, X_valid, y_valid)
        result, sorted_idx = imp_feature.feat_importance()
        print('Top 5 features later ones are more important: ', X_valid.columns[sorted_idx[-5:]])
        #scores_valid.feature_importance_plot(result, data, sorted_idx)

        # Choose the top 8 features and run the predictions again
        feat = X_valid.columns[sorted_idx[-5:]]
        top_8_features = [i for i in feat]
        print('-------------')
        print('Now we choose top 5 features and run the model again')
    
    # This steps returns the fitted model ready for prediction
        clf = model.defaultmodel(X_train[train_df.columns[sorted_idx[-20:]]], y_train)
        print('-------------')
        print("Scores for train data")
        y_pred_train = clf.predict(X_train[train_df.columns[sorted_idx[-20:]]])
        scores_train = PlotsAndScores(y_train, y_pred_train, None)
        scores_train.print_scores()
        print('-------------')
        print("Scores for test data")
        y_pred_valid = clf.predict(X_valid[train_df.columns[sorted_idx[-20:]]])
        scores_valid = PlotsAndScores(y_valid, y_pred_valid, None)
        scores_valid.print_scores()
        print('######')