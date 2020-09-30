import sys
import os
TRAINING_DATA = os.environ.get("TRAINING_DATA")
CLEANED_DATA =os.environ.get("CLEANED_DATA")
from random import randint
import pandas as pd
import numpy as np
from helper.imports import *
from helper.structured  import *
from IPython.display import display
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from helper.plots_and_scores import *
if __name__ ='__main__':

