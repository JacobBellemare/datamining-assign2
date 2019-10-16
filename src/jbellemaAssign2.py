
#A skeleton for implementing K-nearest Neighbor classifier in Python.
## Author: Salem 

import numpy
import random
import time

trainingFile = "irisTraining.txt"
testingFile = "irisTesting.txt"
Xtrain = numpy.loadtxt(trainingFile)
n = Xtrain.shape[0]
d = Xtrain.shape[1]-1
print(n, d)

#Testing .....
Xtest = numpy.loadtxt(testingFile)
nn = Xtest.shape[0] # Number of points in the testing data.

tp = 0 #True Positive
fp = 0 #False Positive
tn = 0 #True Negative
fn = 0 #False Negative

k = 21

#Iterate over all points in testing data
  #For each point find the distances to all the training points.
  #Choose the K points with the smallest distances
  #Assign the class label for the testing point as the majority label of the closes K points.
  #increment TP,FP,FN,TN accordingly, remember the true lable for the ith point is in Xtest[i,d]
testDataPoint = Xtest[0]

for x in range(len(Xtest)):
    testDataPoint = Xtest[x, :]
    distances = []

    for index in Xtrain:
        distance = numpy.linalg.norm(testDataPoint[0: d] - index[0:d])
        distances.append([distance, index[d]])

    distances.sort()
    kneighbors = distances[0: k]
    norm = 0  # increment if class label is 1
    anom = 0  # increment if class label is -1
    for datapoint in kneighbors:
        if (datapoint[1] == 1):
            norm += 1
        else:
            anom += 1
    if(norm > anom):
            predictedClassLabel = 1
    else:
            predictedClassLabel = -1
    actualClassLabel = Xtest[x, d]
    if actualClassLabel == predictedClassLabel:
        if predictedClassLabel == 1:
            tp += 1
        else:
            tn += 1
    else:
        if predictedClassLabel == 1:
            fp += 1
        else:
            fn += 1

print("tp: ", tp)
print("fp: ", fp)
print("tn: ", tn)
print("fn: ", fn)

#calculate accuracy, sensitity, specificity, precision
print("K = ", k)
accuracy = (tp + tn)/(tp + fp + tn + fn)
print("Accuracy: ", accuracy)
sensitivity = tp/(tp + fn)
print("Sensitivity: ", sensitivity)
specificity = tn/(fp + tn)
print("specificity: ", specificity)
precision = tp/(tp + fp)
print("precision: ", precision)
