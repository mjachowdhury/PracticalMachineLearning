# -*- coding: utf-8 -*-

from sklearn import neighbors
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import numpy as np


def question1Part1(train_features, train_labels, test_features, test_labels):
    
    # Explore and plot results for combinations of hyper-parameters (k and distance metric)
    allResults = []
    
    for pVal in range(1, 3):
        for kValue in range(1, 50):
    
            knn = neighbors.KNeighborsClassifier(n_neighbors=kValue, metric="minkowski", p=pVal)
            knn = knn.fit(train_features, train_labels)
            accuracy = knn.score(test_features, test_labels)
            allResults.append(accuracy)
            
        plt.figure(pVal)
        plt.plot(allResults)
    plt.show()


def question1Part2(train_features, train_labels, test_features, test_labels):
    
    # Evaluate Decision Tree model
    decTree = DecisionTreeClassifier()
    decTree = decTree.fit(train_features, train_labels)
    accuracy = decTree.score(test_features, test_labels)
    print ("Decision Tree Accuracy ", accuracy)
    
    # Evaluate Naive Bayes model
    nBayes = GaussianNB()
    nBayes = nBayes.fit(train_features, train_labels)
    accuracy = nBayes.score(test_features, test_labels)
    print ("Naive Bayes Accuracy ", accuracy)
    
    # Evaluate SVM model
    svc = SVC(gamma='auto')
    svc = svc.fit(train_features, train_labels)
    accuracy = svc.score(test_features, test_labels)
    print ("SVM Accuracy ", accuracy)    

    
    
    
    
def main():
    
    # Load dataset into a NumPy array
    cancerTrain = np.genfromtxt("trainingData2.csv", delimiter=",")
    cancerTest = np.genfromtxt("testData2.csv", delimiter=",")
    
   
    # Extract features and label data
    train_features = cancerTrain[:, :-1]
    train_labels = cancerTrain[:, -1]

    test_features = cancerTest[:, :-1]
    test_labels = cancerTest[:, -1]

    from sklearn.preprocessing import StandardScaler
    
    scaler = StandardScaler()
    scaler.fit(train_features)
    
    train_features = scaler.transform(train_features)
    test_features = scaler.transform(test_features)
    
        
    # Evaluate the performance of a kNN
    question1Part1(train_features, train_labels, test_features, test_labels)
    
    # Evaluate alternative models
    question1Part2(train_features, train_labels, test_features, test_labels)
    



main()