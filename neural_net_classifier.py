import numpy as np
import csv
import tensorflow as tf
import keras
from keras.layers import Input, Dense, Dropout
from keras import regularizers, models, optimizers
from keras.models import Sequential

filename_r = 'data//winequality-red.csv'
filename_w = 'data//winequality-white.csv'

n = 4898 # number of white wines
d = 11 # number of columns - 1 (we predict the last column)
data = []
features = []
# let's just look at white wines for now
# with open(filename_r, newline='') as file:
# 	reader = csv.reader(file, delimiter = ';')
# 	for i, row in enumerate(reader):
# 		if i == 0:
# 			features = row
# 		else:
# 			data.append(row)

with open(filename_w, newline='') as file:
	reader = csv.reader(file, delimiter = ';')
	for i, row in enumerate(reader):
		if i == 0:
			features = row
		else:
			data.append(row)

# at this point, our data includes strings and numbers (still scored as strings)
# we need to convert everything to numeric values, then put in an np.array
# for example, change 'FALSE' -> 0, 'TRUE' -> 1, and use some sort of encoding for 'Visitor_Type'

# the last row of our data is QUALITY: let's try to predict whether a wine is above or below average
X = np.zeros((n,d))
Y = np.zeros(n) 
for i in range(n):
	if float(data[i][-1]) >= 6: Y[i] = 1 # good wines are calssified as 1
	else: Y[i] = 0 # bad wines are classified as 0
	for j in range(d):
		X[i,j] = float(data[i][j])

# TODO: should probably rescale/normalize each feature to be on the same scale

num_good = np.sum(Y) 
# Someone like me who doesn't know much about wine and thinks they all taste pretty good would classify each wine as a 1
# This is our 'null model', which would have accuracy ~ num_good/n

#normalize data via max-min rescaling -> puts every entry in [0,1]
new_X = np.zeros((n,d))
for j in range(d):
	col = X[:,j]
	big = float(np.max(col))
	small = float(np.min(col))
	for i in range(n):
		new_X[i,j] = (X[i,j] - small)/(big - small)
print('Original: \n', X[0:3,:])
print('Rescaled: \n', new_X[0:3,:])

# separate into training and testing data
num_train = int(.8 * n)
num_test = n - num_train
X_train = new_X[0:num_train, :]
X_test = new_X[num_train:,:]
Y_train = np.zeros((num_train, 2))
Y_test = np.zeros((num_test, 2))
for i in range(n):
	ind = 0
	if Y[i] == 0: 
		ind = 1
	if i < num_train:
		Y_train[i,ind] = 1
	else:
		Y_test[i-num_train,ind] = 1
# print('Y: \n', Y[0:20])
# print('y train: \n', Y_train[0:20,:].T)

# build a neural network: input layer has d=11 nodes, output layer has 2 nodes
act = 'relu' # the typical activation function we'll use
nn = Sequential()
nn.add(Dense(10, activation=act))
nn.add(Dense(15, activation=act))
nn.add(Dense(15, activation=act))
nn.add(Dense(10, activation=act))
nn.add(Dense(4, activation=act))
nn.add(Dense(2, activation='softmax'))
nn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
nn.fit(x=X_train, y=Y_train, epochs=10, batch_size=32, shuffle=True)
nn.summary()

nn.evaluate(x=X_test, y=Y_test)
predictions = nn.predict(x=X_test)
print(predictions)