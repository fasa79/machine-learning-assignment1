# -*- coding: utf-8 -*-
"""
Created on Sun Dec  6 17:35:36 2020

@author: Fadhluddin bin Sahlan 1817445
Topic: Predicting the type of star using KNN algorithm
Assignment 1 - Machine Learning
"""

import random
import csv
import pylab
import numpy
import copy

"""GET DATA FROM CSV"""

def getData(filename):
    with open(filename) as csvDataSet:
        csvRead = csv.reader(csvDataSet)
        
        for eachdata in csvRead:
            dataset.append(eachdata)
            
    csvDataSet.close()
    
    dataset.pop(0)
    
    return dataset

"""DATA PREPROCESSING"""

def scaleAttrs(vals):
    vals = pylab.array(vals)
    mean = sum(vals)/len(vals)
    sd = numpy.std(vals)
    vals = vals - mean
    return vals/sd

def processData(dataset):
    processedData = numpy.empty((240, 0), float)
    
    #1 Feature Selection (Temperature[0], Luminosity[1], Radius[2], (TARGET)StarType[4], StarColor[5])
    for row in dataset:
        del row[3]
        del row[5]
    
    #2 Data transformation (Change string to float or int and scale them)
    for row in dataset:
        row[4] = row[4].lower()  #Easy to compare
        
        if row[4] == "blue" or row[4] == "blue ":
            row[4] = 0
            
        elif row[4] == "blue white" or row[4] == "blue white " or row[4] == "blue-white":
            row[4] = 1
            
        elif row[4] == "yellowish white" or row[4] == "yellow-white" or row[4] == "white-yellow" or row[4] == "yellowish":
            row[4] = 2
            
        elif row[4] == "white" or row[4] == "white" or row[4] == "whitish":
            row[4] = 3
        
        elif row[4] == "pale yellow orange" or row[4] == "orange-red" or row[4] == "orange":
            row[4] = 4
            
        elif row[4] == "red":
            row[4] = 5
        
        else:
            row[4] = "NaN"
        
        row[3], row[4] = row[4], row[3]   #Swap column target to last column
        
        for i in range(5):
            row[i] = float(row[i])
        
    dataset = numpy.array(dataset)
    for i in range(4):
        temp = scaleAttrs(dataset[:, [i]])
        processedData = numpy.append(processedData, temp, axis = 1)
    
    processedData = numpy.append(processedData, dataset[:, [4]], axis = 1)
    processedData = processedData.tolist()
    
    return processedData

#3 Split Data into training and testing data      
def splitData(dataset, train_data, test_data, ratio):
    size = len(dataset)
    index = 0
    
    random.shuffle(dataset)
    
    while index < size*ratio:
        train_data.append(dataset[index])
        index += 1
    
    while index < size:
        test_data.append(dataset[index])
        index += 1
        
    return train_data, test_data

"""START OF PREPROCESSING THE DATA"""

dataset = []
data = []
train_data = []
test_data = []

dataset = getData("6 class csv.csv")    #GET DATA FROM CSV
data = copy.deepcopy(dataset)
data = processData(dataset)             #PROCESS THE DATA

train_data, test_data = splitData(data, train_data, test_data, 0.8)     #SPLIT DATA INTO TRAINING AND TESTING DATA

with open('Train_data.csv', 'w', newline="") as f:          #output train data to new csv file
    write = csv.writer(f)
    write.writerows(train_data)

with open('Test_data.csv', 'w', newline="") as f:           #output test data to new csv file
    write = csv.writer(f)
    write.writerows(test_data)
        