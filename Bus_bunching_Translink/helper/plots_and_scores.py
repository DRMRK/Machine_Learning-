#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 25 14:30:16 2020

@author: Mukut Ranjan Kalita
"""

from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import precision_recall_curve, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import make_scorer, roc_curve, roc_auc_score, auc
from sklearn.metrics import fbeta_score, accuracy_score
from pprint import pprint
import numpy as np
from sklearn.metrics import plot_confusion_matrix

class plots_and_scores():
    """ 
    Input 
    ---------
    y : actual target values
    y_pred : predicted target  values 
    y_pred_proba : predicted probabilities  
    
    """

    def __init__(self, y, y_pred, y_pred_proba):
        self.y = y
        self.y_pred = y_pred
        self.y_pred_proba = y_pred_proba

    def plot_ROC(self, name):
        self.name = name
        n_classes = 1
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(self.y, self.y_pred)
            roc_auc[i] = auc(fpr[i], tpr[i])
        # Compute micro-average ROC curve and ROC area
        fpr["micro"], tpr["micro"], _ = roc_curve(self.y.ravel(), self.y_pred.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
        plt.figure()
        lw = 2
        plt.plot(fpr[0], tpr[0], color='darkorange',
                 lw=lw, label='ROC curve (area = %0.4f)' % roc_auc[0])
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic %s' % name)
        plt.legend(loc="lower right")
        plt.show()

    def print_scores(self):
        print("Accuracy score: {:.4f}".format(accuracy_score(self.y, self.y_pred)))
        print("MCC score: {:.4f}".format(self.mcc_Metric()))
        print("F-score: {:.4f}".format(self.fbetascore()))

    def fbetascore(self):
        return fbeta_score(self.y, self.y_pred, beta=0.5)

    def mcc_Metric(self):
        cf_matrix = confusion_matrix(self.y, self.y_pred)
        TP = cf_matrix[0][0]
        TN = cf_matrix[1][1]
        FN = cf_matrix[0][1]
        FP = cf_matrix[1][0]
        N = TN + TP + FN + FP
        S = (TP + FN) / N
        P = (TP + FP) / N
        num = (TP / N) - (S * P)
        deno = np.sqrt(P * S * (1 - S) * (1 - P))

        # Need to avoide division by zero
        return self.weird_division(num, deno)

    def display_confusion_matrix(self, filepath, title):

        self.filepath = filepath
        self.title = title
        cm = confusion_matrix(self.y, self.y_pred)
        cmd = ConfusionMatrixDisplay(cm)
        cmd.plot(cmap='gist_ncar')
        fig1=plt.gcf()
        plt.title(self.title)
        plt.show()
        plt.draw()
        fig1.savefig(f'{self.filepath}')



    def weird_division(self, n, d):
        self.n = n
        self.d = d
        return self.n / self.d if self.d else 0

    def precision_recall_vs_threshold(self):
        precisions, recalls, thresholds = precision_recall_curve(self.y, self.y_pred_proba[:, 1])
        plt.figure()
        plt.title("Precision-Recall vs Threshold Chart")
        plt.plot(thresholds, precisions[: -1], "b--", label="Precision")
        plt.plot(thresholds, recalls[: -1], "r--", label="Recall")
        plt.ylabel("Precision, Recall")
        plt.xlabel("Threshold")
        plt.legend(loc="lower left")
        plt.show()


## Some more functions

def mcc_Metric(y, y_pred):
    cf_matrix = confusion_matrix(y, y_pred)
    TP = cf_matrix[0][0]
    TN = cf_matrix[1][1]
    FN = cf_matrix[0][1]
    FP = cf_matrix[1][0]
    N = TN + TP + FN + FP
    S = (TP + FN) / N
    P = (TP + FP) / N
    num = (TP / N) - (S * P)
    deno = np.sqrt(P * S * (1 - S) * (1 - P))

    # Need to avoide division by zero
    return weird_division(num, deno)


def weird_division(n, d):
    return n / d if d else 0
