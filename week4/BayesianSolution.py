# -*- coding: utf-8 -*-


import os
import numpy as np


def buildVocab(path, vocab):
    
    listing = os.listdir(path)
    
    # iterate through each file, extract all words and add to vocab set
    for eachFile in listing:

        f = open(path+eachFile, "r", encoding="utf8")
        allWords = f.read().split()
        vocab.update(allWords)
        f.close()

    print (len(vocab))


def countFrequency(path, posFreqs):
    
    listing = os.listdir(path)
    
    # iterate through each word in each file
    for eachFile in listing:
        
        f = open(path+eachFile, "r", encoding="utf8")
        allWords = f.read().split()
        f.close()
        
        # update the frequency of this word in the word frequency dictionary
        for word in allWords:
            posFreqs[word] += 1 
        

def calculateProbabilities(vocab, probs, freqs):
    
    totalUniqueWords = len(vocab)
    totalClassWords = sum(freqs.values()) 
    
    # for each word calculate the conditional probability of the word 
    # using multi-nomial equation
    for key in freqs:
        probs[key] = (freqs[key]+1)/(totalClassWords + totalUniqueWords)


# In this function we pass in the true class and the false class
def evaluateModelPerformance(trueClassProbs, falseClassProbs, trueClassPrior, falseClassPrior, path, vocab):
    
    listing = os.listdir(path)
    numberOfTestsFiles = len(listing)
    
    numTruePredictions = 0
    
    # iterate through each test file
    for eachFile in listing:
        
        f = open(path+eachFile, "r", encoding="utf8")
        allWords = f.read().split()
        
        probTrueGivenWords = np.log(trueClassPrior)
        probFalseGivenWords = np.log(falseClassPrior)
        
        for word in allWords:
            
            if word in vocab:
                probTrueGivenWords += np.log(trueClassProbs[word])
                probFalseGivenWords += np.log(falseClassProbs[word])
        
        if probTrueGivenWords > probFalseGivenWords:
            numTruePredictions += 1
        f.close()

    print ("Accuracy ", numTruePredictions/numberOfTestsFiles)
        

def calculatePriorProb(POS_TRAIN, NEG_TRAIN):
    
    numberOfPosFiles = len(os.listdir(POS_TRAIN))
    numberofNegFiles = len(os.listdir(NEG_TRAIN))
    totalNumberFiles = numberofNegFiles+numberOfPosFiles
    
    return (numberOfPosFiles/totalNumberFiles),(numberofNegFiles/totalNumberFiles)


def main():
    POS_TRAIN = 'data\\train\\pos\\'
    NEG_TRAIN = 'data\\train\\neg\\'
    POS_TEST = 'data\\test\\pos\\'
    NEG_TEST = 'data\\test\\neg\\'
    
    # create a vocabulary containing all unique words
    # irrespective of class
    vocab = set()
    buildVocab(POS_TRAIN, vocab)
    buildVocab(NEG_TRAIN, vocab)
    
    print ("Size of Vocab",len(vocab))
    
    # create dictionary for positive and negative class that will contain word frequencies
    # dictionary keys are words from vocab
    posFreqs = dict.fromkeys(vocab, 0)
    negFreqs = dict.fromkeys(vocab, 0)
    
    # count the frequency of words in positive and negative training files
    countFrequency(POS_TRAIN, posFreqs)
    countFrequency(NEG_TRAIN, negFreqs)
    
    print ("Number of negative words ", sum(negFreqs.values()) )
    print ("Number of pos words ", sum(posFreqs.values()) )
           
    # create a dictionary to house positive and negative conditional probabilities
    posProbs = dict.fromkeys(vocab, 0)
    negProbs = dict.fromkeys(vocab, 0)
    
    priorPosProb, priorNegProb = calculatePriorProb(POS_TRAIN, NEG_TRAIN)
    
    # calculate conditional probability for each word in positive and negative class
    calculateProbabilities(vocab, posProbs, posFreqs)
    calculateProbabilities(vocab, negProbs, negFreqs)
    
    # evaluate the model for positive test class
    print ("Evaluating Model on Positive Test Files ...")
    evaluateModelPerformance(posProbs, negProbs, priorPosProb, priorNegProb, POS_TEST, vocab)
    
    # evaluate the model for negative test class
    print ("Evaluating Model on Negative Test Files ...")
    evaluateModelPerformance(negProbs, posProbs, priorNegProb, priorPosProb, NEG_TEST, vocab)

main()    

