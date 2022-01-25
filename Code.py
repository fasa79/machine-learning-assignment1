# -*- coding: utf-8 -*-
"""
Created on Thu Dec  3 17:21:13 2020

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

class KNearestNeighbours(object):
    def __init__(self, k):
        self.k = k
        
    @staticmethod
    def euclidean_distance(v1, v2):
        v1, v2 = numpy.array(v1), numpy.array(v2)
        distance = 0
        for i in range (len(v1) - 1):
            distance += (v1[i] - v2[i]) ** 2
        return numpy.sqrt(distance)

    def predict(k, train_set, test_instance):
        distance = []
        for i in range(len(train_set)):
            dist = KNearestNeighbours.euclidean_distance(train_set[i][:-1], test_instance)
            distance.append((train_set[i], dist))
        distance.sort(key=lambda x: x[1])

        neighbours = []
        for i in range(k):
            neighbours.append(distance[i][0])

        classes = {}
        for i in range (len(neighbours)):
            response = neighbours[i][-1]
            if response in classes:
                classes[response] += 1
            else:
                classes[response] = 1

        sorted_classes = sorted(classes.items(), key=lambda x: x[1], reverse=True)

        return sorted_classes[0][0]
    
"""""START OF PROGRAM"""""
#INITIALIZATION
dataset = []
data = []
train_data = []
test_data = []
preds = []

dataset = getData("6 class csv.csv")    #GET DATA FROM CSV
data = copy.deepcopy(dataset)
data = processData(dataset)             #PROCESS THE DATA

train_data, test_data = splitData(data, train_data, test_data, 0.8)     #SPLIT DATA INTO TRAINING AND TESTING DATA

for row in test_data:
    predictors_only = row[:-1]      #remove target of test_data
    prediction = KNearestNeighbours.predict(7, train_data, predictors_only)     #PRODUCE A KNN FROM TRAINING DATA AT THE SAME TIME PREDICT FOR TESTING DATA
    preds.append(prediction)

actual = numpy.array(test_data, dtype = int)[:, -1]      #retrieve the actual target for the test data
preds = numpy.array(preds, dtype = int)
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn import metrics

cnf_mtx = metrics.confusion_matrix(actual, preds)
print(cnf_mtx)

print("Accuracy:",accuracy_score(actual,preds))
print("Precision:",precision_score(actual,preds, pos_label='positive',average='macro'))
print("Recall:",recall_score(actual,preds, pos_label='positive',average='macro'))