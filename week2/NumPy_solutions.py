# -*- coding: utf-8 -*-
"""

MSc in AI
Solutions for NumPy Exercises 

"""

import numpy as np


############ Solutiosn to Question 1 Rainfall #########################

def calculateTotalRainfall(data, startMonth, endMonth):
    
    montlyFilter =  (data[:, 1] >= startMonth) & (data[:, 1] <= endMonth) 
    monthlyData = data[montlyFilter]
    print ("Average total rainfall for month {} to month {} is {}.".format(startMonth, endMonth, np.mean(monthlyData[:,2])))
    
    
def rainfallSolutions():
    
    # load data
    corkData = np.genfromtxt("CorkRainfall.txt")
    dublinData = np.genfromtxt("DublinRainfall.txt")

    # Q1 (i) max and mean of ‘Most Rainfall in a Day’ value
    print ("Question 1 (i)")
    print (np.max(corkData[:,3]))
    print (np.mean(corkData[:,3]))

    # Q1 (ii) Total number of raindays for a specific year
    print ("\n\n Question 1 (ii)")
    uniqueYears = np.unique(corkData[:,0])
    year = int(input("Please enter a year between {} and {}:  ".format(np.min(uniqueYears), np.max(uniqueYears)) ))

    # Extract all rows pertaining to selected year
    yearData = corkData[ corkData[:,0] == year]   
    print ("Total number of raindays ", np.sum(yearData[:, 4]))
    
    
    # Q1 (iii) Calculate wettest month of the year
    print ("\n\n Question 1 (iii)")
    
    maxRainfall = 0.0
    wettestMonth = 1
    uniqueMonths = np.unique(corkData[:,1])
    
    for month in uniqueMonths:
        
        # Extract all rows for specific month
        monthData = corkData[corkData[:, 1] == month] 
        totalRainfall = np.sum(monthData[:,2])
        
        if totalRainfall > maxRainfall:
            maxRainfall = totalRainfall
            wettestMonth = month
    
    print ("Wettest month is {} with a total rainfall value of {}".format(wettestMonth, maxRainfall))
        
    
        
    # Q 1 (iv) Percentage raindays below theshold
    
    print ("\n\n Question 1 (iv)")
    threshold = int(input("Please enter max theshold for number of raindays: "))
    thresholdData = corkData [ corkData[:, 4] <= threshold ]
    
    perctageBelowThreshold = (len(thresholdData)/len(corkData))*100
    print ("Percentage of raindays below threshold {}.".format(perctageBelowThreshold))
    
    
    # Q 1 (v) average ‘total rainfall’ value for the summer months (June, July and August) and the Autumn months (Sept, Oct, Nov)
    
    print ("\n\n Question 1 (v)")
    calculateTotalRainfall(corkData, 6, 8)   
    calculateTotalRainfall(corkData, 9, 11)

    
    # Q 1 (vi) Append dublin and cork data and calculate average raindays
    
    print ("\n\n Question 1 (vi)")
    mergedData = np.append(corkData, dublinData, axis = 0)
    print ("Average number of rain days for Cork and Dublin is {}.".format(np.mean(mergedData[:,4])))
    np.savetxt("DublinCorkRainfall.csv", mergedData, fmt="%f", delimiter=",")
    



############ Solutions to Question 2 Bicycle Dataset #########################


def cacluateMeanUsersHoliday(data, dayType):

    dayTypeSubset = data[data[:,5]==dayType]    
    numUsers = dayTypeSubset[:, 15]
    return (np.mean(numUsers))


def calculateUsersPerMonth(data):
 
    for currentMonth in range(1,13):
        
        # Extract all rows for the current month (currentMonth)
        booleanRowsForMonth = (data[:, 3] == currentMonth)
        dataForMonth = data[booleanRowsForMonth]
    
        print ("Total users for month {} is {}".format(currentMonth, np.sum(dataForMonth[:,13])))
        
        


def analyseTemp(data, minValue, maxValue):
    
    # the temperature values stored in the array are multiplied by 41 
    higherTempCondition = (data[:,9]*41)>=minValue    
    lowerTempCondition = (data[:,9]*41)<=maxValue
    
    # extract all temperature values that satisfying boolean filters above
    subset = data[higherTempCondition & lowerTempCondition]

    # calculate mean number of total users
    meanValue = np.mean(subset[:, 15])

    print ("For temp in range {} to {} the mean number of total rental users was {}".format(minValue, maxValue, meanValue))






def bicycleDataSolutions():
    
    data = np.genfromtxt('bike.csv', delimiter=',')
    
    # Q2 (i) Get average tempature
    print ("\n\n Question 2 (i)")
    print ("Average Temperature is {}".format(np.mean(data[:,9]*41.0)))
    
    
    # Q2 (ii) 
    print ("\n\n Question 2 (ii)")
    print ("Mean number of non-holiday users {}".format(cacluateMeanUsersHoliday(data, 0)))
    print ("Mean number of holiday users {}".format(cacluateMeanUsersHoliday(data, 1)))
 
    # Q2 (iii) Mean number of casual users per month
    print ("\n\n Question 2 (iii)")
    calculateUsersPerMonth(data)
    
    # Q2 (iv) Mean number of total users per month
    print ("\n\n Question 2 (iv)")
    for temp in range(1, 40, 5):
        analyseTemp(data, temp, temp+4)
        






def main():
    
    # Solutions to Question 1
    rainfallSolutions()
    
    # Solutions to Question 2   
    bicycleDataSolutions()
    


if __name__ == "__main__":
    main()