# -*- coding: utf-8 -*-
"""
Created on Mon Nov  9 15:24:36 2020

@author: Ted.Scully
"""


import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import Lasso


def main():
    
    trainAll = np.genfromtxt("trainingData.csv", delimiter=",")
    testAll = np.genfromtxt("testData.csv", delimiter=",")
    
    # Extract feature data 
    train_features = trainAll[:, :-1]
    train_labels = trainAll[:, -1]
    
    test_features = testAll[:, :-1]
    test_labels = testAll[:, -1]
    
    
    reg_model = Lasso()
    reg_model.fit(train_features, train_labels)
    print("R2 result for Lasso without removing outliers: ", reg_model.score(test_features, test_labels))
    
    # Create an isolation forest to remove mutlivariate outliers
    clf = IsolationForest(contamination = 0.01)
    clf.fit(train_features)
        
    # Predict returns an array contains 1 (not outlier) and -1 (outlier) values 
    results = clf.predict(train_features)
    
    # Exact only non-outlier instances
    normal_features = train_features[results == 1]
    normal_labels = train_labels[results == 1]
    
    # Rerun the Lasso regression model
    reg_model = Lasso()
    reg_model.fit(normal_features, normal_labels)
    print("R2 result for Lasso after removal of outliers: ",reg_model.score(test_features, test_labels))
    


main()

