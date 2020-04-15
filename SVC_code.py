# Python program that can be executed to report whether particular
# python packages are available on the system.

import math
import os
import sys




#SVC_clf = SVC()
#SVC.fit(features, labels)
#SVC.predict(test_features)

import pandas as pd
import numpy as np
import csv

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.svm import SVC


SVC_clf = SVC()

filename_r = 'data//winequality-red.csv'
filename_w = 'data//winequality-white.csv'


n = 4898    #number of white wines
d = 12     #number of features
data = []
features = []

with open(filename_w, newline = '') as file:
    reader = csv.reader(file, delimiter = ';')
    for i, row in enumerate(reader):
        if i == 0:
            features = row
        else:
            data.append(row)
            
X = np.zeros((n,d-1))
Y = np.zeros(n)
for i in range(n):
    if float(data[i][-1]) >= 6: 
        Y[i] = 1
    else:
        Y[i] = 0
    for j in range(d-1):
        X[i,j] = float(data[i][j])

num_good = np.sum(Y)

num_train = int(.8 * n)
num_test = n - num_train
X_train = X[0:num_train, :]
X_test = X[num_train:, :]
Y_train = Y[0:num_train]
Y_test = Y[num_train:]

print('Features: ', features)
print('Data: ', X_train[0:5,:])
print('Labels: ', Y[0:5])

#work on normalizing data

from sklearn import preprocessing

std_scale = preprocessing.StandardScaler().fit(X_train)
X_train_std = std_scale.transform(X_train)
X_test_std = std_scale.transform(X_test)
        

SVC_clf.fit(X_train_std, Y_train)

#specificity / sensitivity
#false positive rate

'''
SVC_prediction = SVC_clf.predict(X_test)
print(SVC_prediction)
print(np.sum(Y_test))
print(np.sum(SVC_prediction))
print(accuracy_score(SVC_prediction, Y_test))
print(precision_score(SVC_prediction, Y_test))
print(recall_score(SVC_prediction, Y_test))
print(confusion_matrix(SVC_prediction, Y_test))
print(classification_report(SVC_prediction, Y_test))
'''


SVC_prediction = SVC_clf.predict(X_test_std)
print(SVC_prediction)
print(np.sum(Y_test))
print(np.sum(SVC_prediction))
print(accuracy_score(SVC_prediction, Y_test))
print(precision_score(SVC_prediction, Y_test))
print(recall_score(SVC_prediction, Y_test))
print(confusion_matrix(SVC_prediction, Y_test))
print(classification_report(SVC_prediction, Y_test))




 