
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
print n, d

#Testing .....
Xtest = numpy.loadtxt(testingFile)
nn = Xtest.shape[0] # Number of points in the testing data.

tp = 0 #True Positive
fp = 0 #False Positive
tn = 0 #True Negative
fn = 0 #False Negative

#Iterate over all points in testing data
  #For each point find the distances to all the training points.
  #Choose the K points with the smallest distances
  #Assign the class label for the testing point as the majority label of the closes K points.
  #increment TP,FP,FN,TN accordingly, remember the true lable for the ith point is in Xtest[i,d]

#}

#Calculate all the measures required..
 
