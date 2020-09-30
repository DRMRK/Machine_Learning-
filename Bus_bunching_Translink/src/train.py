# We use this to analyze the data cleaned in clean_data.py

import sys
import os
from random import randint
import pandas as pd
import numpy as np
from helper.imports import *
from helper.structured import *
from IPython.display import display
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from helper.plots_and_scores import *

TRAINING_DATA = os.environ.get("TRAINING_DATA")
CLEANED_DATA = os.environ.get("CLEANED_DATA")

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
    X_train, X_test_valid, y_train, y_test_valid = \
        train_test_split(X, y, random_state=1, test_size = 0.20,
                         stratify=y)
    X_test, X_valid, y_test, y_valid = \
        train_test_split(X_test_valid, y_test_valid,
                         random_state=1, test_size=0.50,
                         stratify=y_test_valid)
    # Now the data is ready for modelling

