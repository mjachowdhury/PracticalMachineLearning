# -*- coding: utf-8 -*-



from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ShuffleSplit
from sklearn import ensemble
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
from sklearn.datasets import load_digits



## Solution to Part A

def question1():

    digits = datasets.load_digits()
    X = digits.data
    y = digits.target
    
    print (X.shape)
    
    estimator = DecisionTreeClassifier(max_depth = 2)
    
    
    title = "Learning Curve Iris (NB)"
    plt.title(title)
    ylim=(0.0, 1.01)
    
    
    plt.ylim(ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
        
    train_sizes=np.linspace(.05, 1.0, 10)
    custom_cv = ShuffleSplit(n_splits=100, test_size=0.1, random_state=0)
    train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, train_sizes=train_sizes, cv = custom_cv)
            
    # remember axis = 1 operates on the horizontal axis and calculates the mean below for       # each row. Each row corresponds to one tick and each value returned in a row refer to the 
    # accuracy for that fold. 
    train_scores_mean = np.mean(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    plt.grid()
    
    
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")
    
    plt.legend(loc="best")
    plt.show()



### Solution to PART B


def question2():
    
    dfData = pd.read_csv('sampleDataset.csv')
    
    nData = dfData.values
    
    X = nData[:, :-1]
    y = nData[:, len(nData[0])-1]
    
    estimator = GaussianNB() 
    
    
    title = "Learning Curve"
    plt.title(title)
    ylim=(0.0, 1.01)
    
    
    plt.ylim(ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
        
    train_sizes=np.linspace(.05, 1.0, 10)
    custom_cv = ShuffleSplit(n_splits=30, test_size=0.1, random_state=0)
    train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, train_sizes=train_sizes, cv = custom_cv)
            
    # remember axis = 1 operates on the horizontal axis and calculates the mean below for       # each row. Each row corresponds to one tick and each value returned in a row refer to the 
    # accuracy for that fold. 
    train_scores_mean = np.mean(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    plt.grid()
    
    
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")
    
    plt.legend(loc="best")
    plt.show()



def main():
    question1()
    question2()
    

main()