# -*- coding: utf-8 -*-
"""
Created on Sun Dec  6 17:35:35 2020

@author: Fadhluddin bin Sahlan 1817445
Topic: Predicting the type of star using KNN algorithm
Assignment 1 - Machine Learning
"""
import csv
import numpy
import pandas as pd

def getData(data, filename):
    with open(filename) as csvDataSet:
        csvRead = csv.reader(csvDataSet)
        
        for eachdata in csvRead:
            data.append(eachdata)
            
    csvDataSet.close()
    
    for row in data:
        for i in range(5):
            row[i] = float(row[i])
            
    return data

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
  
#PARAMETER K
k = 1

train_data = []
test_data = []
preds = []


train_data = getData(train_data, "Train_data.csv")
test_data = getData(test_data, "Test_data.csv")

for row in test_data:
    predictors_only = row[:-1]      #remove target of test_data
    prediction = KNearestNeighbours.predict(k, train_data, predictors_only)     #PRODUCE A KNN FROM TRAINING DATA AT THE SAME TIME PREDICT FOR TESTING DATA
    preds.append(prediction)

actual = numpy.array(test_data, dtype = int)[:, -1]      #retrieve the actual target for the test data
preds = numpy.array(preds, dtype = int)
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn import metrics

print("Actual: ", actual)
print("\nPreds : ", preds)
cnf_mtx = metrics.confusion_matrix(actual, preds)
print("\nConfusion Matrix:\n", cnf_mtx)

print("\nAccuracy:",accuracy_score(actual,preds))
print("\nPrecision:",precision_score(actual,preds, pos_label='positive',average='macro'))
print("\nRecall:",recall_score(actual,preds, pos_label='positive',average='macro'))

# import required modules
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class_names=[0,1] # name  of classes
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
# create heatmap
sns.heatmap(pd.DataFrame(cnf_mtx), annot=True, cmap="YlGnBu" ,fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')