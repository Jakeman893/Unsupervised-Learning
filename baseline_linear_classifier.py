__author__ = 'm.bashari'
import numpy as np
from sklearn import linear_model

def classify(X, y):
    clf = linear_model.LogisticRegressionCV()
    clf.fit(X, y)
    return clf
